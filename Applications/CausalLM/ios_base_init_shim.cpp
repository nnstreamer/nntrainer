// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file ios_base_init_shim.cpp
 * @brief Link-time compatibility shim for the vendored tokenizers prebuilt
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <iostream>

// lib/libtokenizers_c.a (tracked prebuilt of the tokenizers_c_win crate,
// built with GCC 13+) references std::ios_base_library_init() -- the GCC 13+
// libstdc++ replacement for <iostream>'s static-init object. Ubuntu 22.04's
// libstdc++ (GCC 12) predates it, so every executable that links
// libcausallm.so (which absorbs the archive) fails with "undefined reference
// to std::ios_base_library_init()". Provide the definition here with the
// original effect: run ios_base::Init exactly once so std::cout/cerr are
// usable from the tokenizer's translation units. Kept out of any header --
// GCC 13+ toolchains that provide the real symbol in libstdc++.so will
// simply resolve to whichever definition the link picks; the semantics are
// identical and ios_base::Init is refcounted.
namespace std {
__attribute__((visibility("default"))) void ios_base_library_init() {
  static ios_base::Init s_iostream_init;
}
} // namespace std
