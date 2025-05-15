//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/*!
 * Copyright (c) 2023 by Contributors
 * \file tokenizers_capi.h
 * \brief C binding to tokenizers rust library
 */
#ifndef QUALLA_TOKENIZERS_CAPI_H
#define QUALLA_TOKENIZERS_CAPI_H

// The C API
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef void* TokenizerHandle;

TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);
void tokenizers_encode(TokenizerHandle handle, const char* data, size_t len, int add_special_token);
void tokenizers_decode(
        TokenizerHandle handle,
        const uint32_t* data,
        size_t          len,
        int             skip_special_token
);
void tokenizers_get_decode_str(TokenizerHandle handle, const char** data, size_t* len);
void tokenizers_get_encode_ids(TokenizerHandle handle, const uint32_t** id_data, size_t* len);
void tokenizers_free(TokenizerHandle handle);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZERS_CAPI_H
