// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    image.cc
 * @date    28 Feb 2023
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple image converter for android
 *
 */

#include "image.h"

std::unique_ptr<Image> ImageFactory::FromFd(int fd) {
  AImageDecoder *decoder;
  int result = AImageDecoder_createFromFd(fd, &decoder);
  if (result != ANDROID_IMAGE_DECODER_SUCCESS) {
    return nullptr;
  }

  auto decoder_cleanup = [&decoder]() { AImageDecoder_delete(decoder); };

  const AImageDecoderHeaderInfo *header_info =
    AImageDecoder_getHeaderInfo(decoder);
  int bitmap_format =
    AImageDecoderHeaderInfo_getAndroidBitmapFormat(header_info);
  if (bitmap_format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    decoder_cleanup();
    return nullptr;
  }
  constexpr int kChannels = 4;
  int width = AImageDecoderHeaderInfo_getWidth(header_info);
  int height = AImageDecoderHeaderInfo_getHeight(header_info);

  size_t stride = AImageDecoder_getMinimumStride(decoder);
  std::unique_ptr<Image> image_ptr =
    std::make_unique<Image>(width, height, kChannels, stride);

  size_t size = width * height * kChannels;
  int decode_result =
    AImageDecoder_decodeImage(decoder, image_ptr->pixels(), stride, size);
  if (decode_result != ANDROID_IMAGE_DECODER_SUCCESS) {
    decoder_cleanup();
    return nullptr;
  }

  decoder_cleanup();
  return image_ptr;
}
