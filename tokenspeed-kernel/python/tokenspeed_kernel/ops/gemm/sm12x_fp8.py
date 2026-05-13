# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""Public SM12x FP8 GEMM helper wrappers."""

from tokenspeed_kernel.thirdparty.cuda.sm12x_fp8 import (
    sm12x_fp8_weight_gemv_ue8m0,
    sm12x_mxfp8_block128_quant_dequant_ue8m0,
    sm12x_mxfp8_block128_quantize,
)

__all__ = [
    "sm12x_fp8_weight_gemv_ue8m0",
    "sm12x_mxfp8_block128_quant_dequant_ue8m0",
    "sm12x_mxfp8_block128_quantize",
]
