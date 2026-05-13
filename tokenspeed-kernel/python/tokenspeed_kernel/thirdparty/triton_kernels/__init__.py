# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import importlib

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton


def _optional_import(module_name: str, *, required: bool = False) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_requested_module = exc.name == module_name or (
            exc.name is not None
            and (
                exc.name.startswith(f"{module_name}.")
                or module_name.startswith(f"{exc.name}.")
            )
        )
        if missing_requested_module and not required:
            return
        raise
    except ImportError:
        if required:
            raise
        return


with redirect_triton_to_tokenspeed_triton():
    for _module_name in (
        "triton_kernels",
        "triton_kernels.matmul",
        "triton_kernels.matmul_details",
        "triton_kernels.matmul_details.opt_flags",
        "triton_kernels.matmul_ogs",
        "triton_kernels.numerics",
        "triton_kernels.numerics_details",
        "triton_kernels.routing",
        "triton_kernels.swiglu",
        "triton_kernels.tensor",
        "triton_kernels.tensor_details",
        "triton_kernels.tensor_details.layout",
        "triton_kernels.topk",
        "triton_kernels.topk_details",
    ):
        _optional_import(_module_name, required=(_module_name == "triton_kernels"))
