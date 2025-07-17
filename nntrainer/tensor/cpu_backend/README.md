# cpu backend

## Overview
```
...
tensor
├─── cpu_backend
│   └─── cpu_backend.h (has all external function interfaces for tensor operations)
│   └─── arm
│      └─── arm_compute_backend : copy of `cpu_backend.h`. selectively uses functions from `cblas_interface`, `neon_impl`, `fallback_internal`
│      └─── neon_impl : custom implementation based on neon SIMD intrinsic
│      └─── neon_impl_fp16 (For armv8.2+)
│      └─── armv7_neon (For armv7l)
│           ...
│   └─── x86
│      └─── x86_compute_backend : copy of `cpu_backend.h`. selectively uses functions from `cblas_interface`, `avx2_impl`, `fallback_internal`
│      └─── avx2_impl : custom implementation based on avx2 SIMD intrinsic
│           ...
│   └─── fallback
│      └─── fallback ( !x86 & !arm ) : copy of `cpu_backend.h`. uses ALL functions from `fallback_internal`
│      └─── fallback_internal : all raw implementations without SIMD or external lib
│           ...
│   └─── cblas_interface
│      └─── cblas_interface : all cblas-related function interfaces, and params
...
```
## Basic guidelines for developers
1. If you are considering custom implementation using SIMD intrinsic for interested target HW architecture, you can directly implement under `neon_impl`, `avx2_impl`, or add a new folder for the target architecture / SIMD ISA.
2. If you are considering introducing external library (e.g., eigen3, XNNPACK, etc.), add a new folder like `cblas_interface` and refer to it.
