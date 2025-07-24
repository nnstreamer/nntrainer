// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    image.h
 * @date    28 Feb 2023
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple image converter for android
 *
 */

#include <memory>

#include <android/imagedecoder.h>
#include <assert.h>

/**
 * @class   Image Class
 * @brief   Image Class
 */
class Image {
public:
  friend class ImageFactory;
  /**
   * @brief Image Ojbect with ImageFactory from Android
   *
   * @param width width of image
   * @param height height of image
   * @param channel channel of image
   * @param stride stride of image
   */
  Image(int width, int height, int channels, int stride) :
    width_(width),
    height_(height),
    channels_(channels),
    stride_(stride) {
    this->pixels_ = std::make_unique<uint8_t[]>(width * height * channels);
  }

  /**
   * @brief get pixel data
   *
   * @param x position of width
   * @param y position of height
   * @param c position of channel
   * @return pixel pointer
   */
  uint8_t get_pixel(int x, int y, int c) const {
    uint8_t *pixel = this->pixels_.get() + (y * stride_ + x * 4 + c);
    return *pixel;
  }

  /**
   * @brief get width
   * @return width
   *
   */
  int width() const { return this->width_; }

  /**
   * @brief get height
   * @return height
   *
   */
  int height() const { return this->height_; }

  /**
   * @brief get channel
   * @return channel
   *
   */
  int channels() const { return this->channels_; }
  /**
   * @brief get stride
   * @return stride
   *
   */
  int stride() const { return this->stride_; }

  /**
   * @brief get pixel pointer
   * @return pixel pointer
   *
   */
  uint8_t *pixels() { return this->pixels_.get(); }

private:
  std::unique_ptr<uint8_t[]> pixels_;
  const int width_;
  const int height_;
  const int channels_;
  const int stride_;
};

/**
 * @class   ImageFactory Class
 * @brief   ImageFactory Class
 */
class ImageFactory {
public:
  /**
   * @brief Android FromFd interface
   *
   * @param fd file fd
   */
  static std::unique_ptr<Image> FromFd(int fd);
};
