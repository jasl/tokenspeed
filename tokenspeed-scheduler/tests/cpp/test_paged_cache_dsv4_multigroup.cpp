// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

// Coverage: DeepSeek V4-shaped multi-group paged cache with BOTH families
// required (build_v4_cache_specs ordering: a State group is the FIRST required
// group). Regressions covered:
//  1. Committing across boundary nodes whose State segments were stripped by
//     the superseded-state release must refill them and keep every group's
//     commit cursor in lockstep. Pre-fix this desynced the State cursors via
//     commitTerminalContinuationSnapshot and crashed with
//     "AdoptSnapshotSegment: snapshot segment size mismatch".
//  2. The superseded-state release must retain the MOST RECENT superseded
//     boundary so an identical resend of an aligned prompt still hits at
//     key = N - LCM.
//  3. A late catch-up commit (live sliding window already past the boundary)
//     must omit the State group from the snapshot instead of publishing a
//     ragged/empty segment.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "resource/allocator/owned_pages.h"
#include "resource/allocator/page_allocator.h"
#include "resource/allocator/paged_cache_group.h"
#include "resource/hybrid_prefix_cache/hybrid_prefix_cache.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/paged_cache_snapshot.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "resource/types.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {
namespace {

// DSv4 shapes: KV page 64, history alignment LCM = 256; state windows 128/8.
class PagedCacheDsv4MultiGroupTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kPageSize = 64;
    static constexpr std::int32_t kLcm = 256;
    static constexpr const char* kSwaKv = "swa_kv";
    static constexpr const char* kC4aState = "c4a_state";
    static constexpr const char* kC4aKv = "c4a_kv";
    static constexpr const char* kC128aState = "c128a_state";
    static constexpr const char* kC128aKv = "c128a_kv";
    static constexpr const char* kC4aIdxState = "c4a_idx_state";

    void SetUp() override {
        device_alloc_ = std::make_unique<PageAllocator>(kPageSize, /*total_pages=*/512);
        kv_cache_ = std::make_unique<KVPrefixCache>(device_alloc_.get(), /*host=*/nullptr);
        hybrid_ = std::make_unique<HybridPrefixCache>(*kv_cache_, /*mamba=*/nullptr,
                                                      /*mamba_chunk_size=*/0);

        // build_v4_cache_specs order: swa_kv (State) FIRST, then per-ratio
        // {compressor_state (State), compressed_kv (History)}, then the c4a
        // indexer state. The State-first ordering is load-bearing: it is what
        // exposed the canonical-commit-cursor desync.
        RegisterGroup(kSwaKv, /*rows=*/64, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
                      /*window=*/128, PagedCacheGroupFamily::State, /*total_pages=*/256);
        RegisterGroup(kC4aState, /*rows=*/4, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
                      /*window=*/8, PagedCacheGroupFamily::State, /*total_pages=*/1024);
        RegisterGroup(kC4aKv, /*rows=*/64, /*stride=*/4, PagedCacheGroupConfig::Retention::FullHistory,
                      /*window=*/std::nullopt, PagedCacheGroupFamily::History, /*total_pages=*/64);
        RegisterGroup(kC128aState, /*rows=*/8, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
                      /*window=*/128, PagedCacheGroupFamily::State, /*total_pages=*/1024);
        RegisterGroup(kC128aKv, /*rows=*/2, /*stride=*/128, PagedCacheGroupConfig::Retention::FullHistory,
                      /*window=*/std::nullopt, PagedCacheGroupFamily::History, /*total_pages=*/64);
        RegisterGroup(kC4aIdxState, /*rows=*/4, /*stride=*/1, PagedCacheGroupConfig::Retention::SlidingWindow,
                      /*window=*/8, PagedCacheGroupFamily::State, /*total_pages=*/1024);

        std::unordered_map<std::string, std::int32_t> sliding{
            {kSwaKv, 128}, {kC4aState, 8}, {kC128aState, 128}, {kC4aIdxState, 8}};
        hybrid_->EnablePagedCacheAdjunct({kSwaKv, kC4aState, kC4aKv, kC128aState, kC128aKv, kC4aIdxState},
                                         std::move(sliding));
        kv_cache_->GetDeviceManager().SetEvictionCallback([this](TreeNode* node) { hybrid_->OnKVEvict(node); });
    }

    TreeNode* InsertDeviceTokens(std::int32_t raw_tokens, token_t token_start = 1) {
        const std::int32_t num_pages = raw_tokens / kPageSize;
        auto tokens = MakeAlignedTokens(num_pages, kPageSize, token_start);
        OwnedPages pages = device_alloc_->Allocate(num_pages);
        auto res = kv_cache_->Insert<ResourceType::Device>(tokens, /*prefix_pages=*/{}, std::move(pages),
                                                           /*page_hashes=*/{}, /*start_node=*/nullptr);
        return res.last_node;
    }

    // Prefill/decode-style step: extend the chain to `end`, hold the request's
    // path lock (RefCount == 1, like a live request's DeviceNodeRef), acquire
    // the chunk, then commit. Returns the new terminal.
    TreeNode* ExtendAndCommit(const std::string& request_id, std::int32_t begin, std::int32_t end,
                              std::unique_ptr<DeviceNodeRef>& lock, token_t token_start = 1) {
        TreeNode* terminal = InsertDeviceTokens(end, token_start);
        EXPECT_NE(terminal, nullptr);
        auto new_lock = std::make_unique<DeviceNodeRef>(terminal);
        lock = std::move(new_lock);
        hybrid_->AcquireForRequest(request_id, begin, end);
        hybrid_->CommitChunk(request_id, terminal);
        return terminal;
    }

    TreeNode* BoundaryNode(TreeNode* descendant, std::int32_t depth) {
        return kv_cache_->GetRadixTree().SplitAt(descendant, depth);
    }

    static bool HasStateGroups(const TreeNode* node) {
        const PagedCacheSnapshot* snap = node->GetPagedCacheSnapshot();
        return snap != nullptr && snap->IsCompleteFor(PagedCacheGroupFamily::State);
    }

    std::unique_ptr<PageAllocator> device_alloc_;
    std::unique_ptr<KVPrefixCache> kv_cache_;
    std::unique_ptr<HybridPrefixCache> hybrid_;

private:
    void RegisterGroup(std::string group_id, std::int32_t rows_per_page, std::int32_t stride,
                       PagedCacheGroupConfig::Retention retention, std::optional<std::int32_t> window,
                       PagedCacheGroupFamily family, std::int32_t total_pages) {
        PagedCacheGroupConfig cfg{};
        cfg.group_id = std::move(group_id);
        cfg.rows_per_page = rows_per_page;
        cfg.entry_stride_tokens = stride;
        cfg.total_pages = total_pages;
        cfg.retention = retention;
        cfg.sliding_window_tokens = window;
        cfg.family = family;
        hybrid_->RegisterPagedCacheGroup(std::make_unique<PagedCacheGroupAllocator>(cfg));
    }
};

}  // namespace

// Regression for the AdoptSnapshotSegment "snapshot segment size mismatch"
// crash: request A primes a chain and decodes past several boundaries so the
// superseded-state release strips interior State segments. Request B then
// re-commits along that chain. Pre-fix, adoption failed on every stripped
// node (warning + stuck cursor) while commitTerminalContinuationSnapshot kept
// advancing the State tables' cursors at each aligned terminal; the canonical
// (State-first) cursor ran ahead of the History tables until adoption of a
// State-complete node threw. Post-fix, stripped boundaries are refilled from
// the live tables and every cursor advances in lockstep.
TEST_F(PagedCacheDsv4MultiGroupTest, CommitOverStrippedBoundariesRefillsAndStaysInLockstep) {
    // Request A: chunked prefill to 2048, then decode to 2432 (page-granular
    // publishes). The aligned commits at 2048 and 2304 run the superseded
    // release, stripping interior boundaries.
    std::unique_ptr<DeviceNodeRef> a_lock;
    TreeNode* a_terminal = nullptr;
    for (std::int32_t end = kLcm; end <= 2048; end += kLcm) {
        a_terminal = ExtendAndCommit("A", end - kLcm, end, a_lock);
    }
    for (std::int32_t end = 2048 + kPageSize; end <= 2432; end += kPageSize) {
        a_terminal = ExtendAndCommit("A", end - kPageSize, end, a_lock);
    }

    TreeNode* n256 = BoundaryNode(a_terminal, 256);
    TreeNode* n1792 = BoundaryNode(a_terminal, 1792);
    TreeNode* n2048 = BoundaryNode(a_terminal, 2048);
    TreeNode* n2304 = BoundaryNode(a_terminal, 2304);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n1792, nullptr);
    ASSERT_NE(n2048, nullptr);
    ASSERT_NE(n2304, nullptr);

    // Superseded release ran at chunk depths 2048 and 2304: old interior
    // boundaries are History-only; the most recent superseded boundary (2048)
    // and the non-superseded 2304 keep their State segments.
    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    EXPECT_TRUE(n256->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(HasStateGroups(n256));
    EXPECT_FALSE(HasStateGroups(n1792));
    EXPECT_TRUE(HasStateGroups(n2048));
    EXPECT_TRUE(HasStateGroups(n2304));

    hybrid_->ReleaseRequest("A");
    a_lock.reset();

    // Request B: identical 2304-token prompt walks A's chain. All State
    // segments below 1792 are stripped, so the match falls back to no paged
    // hit and B commits cold over A's existing snapshots. Pre-fix this
    // sequence raised std::invalid_argument("...snapshot segment size
    // mismatch") once the desynced cursor met a State-complete node.
    RadixTree& tree = kv_cache_->GetRadixTree();
    for (std::int32_t end = kLcm; end <= 2304; end += kLcm) {
        hybrid_->AcquireForRequest("B", end - kLcm, end);
        TreeNode* b_terminal = tree.SplitAt(a_terminal, end);
        ASSERT_NE(b_terminal, nullptr);
        ASSERT_NO_THROW(hybrid_->CommitChunk("B", b_terminal)) << "boundary " << end;
    }

    // Lockstep held: all nine 256-token history segments were adopted.
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("B", kC4aKv).size(), 9u);
    EXPECT_EQ(hybrid_->GetRequestPagedCachePageIds("B", kC128aKv).size(), 9u);

    // Stripped boundaries were refilled from B's live tables on the way.
    EXPECT_TRUE(HasStateGroups(n256));
    EXPECT_TRUE(HasStateGroups(n1792));

    // The refilled chain is immediately usable for a third request.
    auto match = hybrid_->Match(MakeAlignedTokens(2304 / kPageSize, kPageSize, 1));
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 2304);
    EXPECT_EQ(match.paged_cache.history_hit_tokens, 2304);

    hybrid_->ReleaseRequest("B");
}

// Task-2 retention: the superseded-state release keeps the MOST RECENT
// superseded boundary (releasing only strictly older ones), so an identical
// resend of an aligned prompt — whose radix walk stops one page short of the
// full prompt — hits at key = N - LCM. The retained boundary advances with
// decode, so memory growth stays bounded at one extra window per chain.
TEST_F(PagedCacheDsv4MultiGroupTest, SupersededReleaseRetainsMostRecentBoundary) {
    std::unique_ptr<DeviceNodeRef> a_lock;
    TreeNode* a_terminal = nullptr;
    for (std::int32_t end = kLcm; end <= 2048; end += kLcm) {
        a_terminal = ExtendAndCommit("A", end - kLcm, end, a_lock);
    }

    TreeNode* n1536 = BoundaryNode(a_terminal, 1536);
    TreeNode* n1792 = BoundaryNode(a_terminal, 1792);
    TreeNode* n2048 = BoundaryNode(a_terminal, 2048);
    ASSERT_NE(n1536, nullptr);
    ASSERT_NE(n1792, nullptr);
    ASSERT_NE(n2048, nullptr);

    // At chunk depth 2048 the superseded set is {..., 1536, 1792}; 1792 is the
    // most recent and must survive, strictly older boundaries are released.
    EXPECT_FALSE(HasStateGroups(n1536));
    EXPECT_TRUE(HasStateGroups(n1792));
    EXPECT_TRUE(HasStateGroups(n2048));

    // Identical aligned-2048 resend: the walk covers (2048 - 1) tokens ->
    // 1984, so the deepest reachable boundary is 1792. Pre-fix it was
    // stripped and the resend matched ZERO cached tokens.
    auto resend = hybrid_->Match(MakeAlignedTokens(1984 / kPageSize, kPageSize, 1));
    EXPECT_EQ(resend.paged_cache.prefix_len_tokens, 1792);
    EXPECT_EQ(resend.paged_cache.history_hit_tokens, 1792);
    EXPECT_FALSE(resend.paged_cache.per_group_page_ids.at(kC128aState).empty());

    // Decode past the next boundary: the retained boundary advances to 2048
    // and the previously retained 1792 is released — exactly one extra
    // retained boundary per chain.
    for (std::int32_t end = 2048 + kPageSize; end <= 2304; end += kPageSize) {
        a_terminal = ExtendAndCommit("A", end - kPageSize, end, a_lock);
    }
    EXPECT_FALSE(HasStateGroups(n1792));
    EXPECT_TRUE(HasStateGroups(n2048));
    EXPECT_TRUE(HasStateGroups(BoundaryNode(a_terminal, 2304)));

    hybrid_->ReleaseRequest("A");
    a_lock.reset();

    // Resend of the (now 2304-token) prompt hits the retained 2048 boundary.
    auto resend2 = hybrid_->Match(MakeAlignedTokens(2240 / kPageSize, kPageSize, 1));
    EXPECT_EQ(resend2.paged_cache.prefix_len_tokens, 2048);
}

// A late catch-up commit — the live sliding window (ReleaseSkipped at the
// current position) has already advanced past an old boundary's window —
// must not publish ragged or empty State segments. The State group is left
// out of that boundary's snapshot (capping match depth there) while History
// commits normally and all commit cursors stay in lockstep.
TEST_F(PagedCacheDsv4MultiGroupTest, LateCatchUpCommitOmitsStateInsteadOfRaggedSegments) {
    // No DeviceNodeRef on the chain: with RefCount == 0 the superseded-state
    // release cannot run, so pre-fix EMPTY segments would survive on the early
    // boundaries and this test would catch them (present-but-empty groups
    // made IsCompleteFor(State) true and poisoned matches/adoptions).
    TreeNode* terminal = InsertDeviceTokens(1024, /*token_start=*/50'000);
    ASSERT_NE(terminal, nullptr);

    // One big op whose first position is deep into the sequence: the sliding
    // tables start at the live window (page base ~ (900 - w + 1) / rpp),
    // far past the windows of boundaries 256/512/768.
    hybrid_->AcquireForRequest("R", /*first_raw_position_of_op=*/900,
                               /*target_raw_tokens_exclusive=*/1024);
    ASSERT_NO_THROW(hybrid_->CommitChunk("R", terminal));

    TreeNode* n256 = BoundaryNode(terminal, 256);
    TreeNode* n1024 = BoundaryNode(terminal, 1024);
    ASSERT_NE(n256, nullptr);
    ASSERT_NE(n1024, nullptr);

    // Early boundary: History committed, State omitted (no empty segments).
    ASSERT_TRUE(n256->HasPagedCacheSnapshot());
    const PagedCacheSnapshot* snap256 = n256->GetPagedCacheSnapshot();
    EXPECT_TRUE(snap256->IsCompleteFor(PagedCacheGroupFamily::History));
    EXPECT_FALSE(snap256->IsCompleteFor(PagedCacheGroupFamily::State));
    EXPECT_EQ(snap256->groups.find(kSwaKv), snap256->groups.end());
    EXPECT_EQ(snap256->groups.find(kC4aState), snap256->groups.end());

    // The final boundary still lies within every live window: fully committed.
    ASSERT_TRUE(n1024->HasPagedCacheSnapshot());
    EXPECT_TRUE(n1024->GetPagedCacheSnapshot()->IsCompleteFor(PagedCacheGroupFamily::State));

    // The chain is usable at its State-complete depth.
    auto match = hybrid_->Match(MakeAlignedTokens(1024 / kPageSize, kPageSize, 50'000));
    EXPECT_EQ(match.paged_cache.prefix_len_tokens, 1024);

    hybrid_->ReleaseRequest("R");
}

}  // namespace tokenspeed::test
