// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    dataloader.cpp
 * @date    08 Sept 2021
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple nntrainer implementaiton with JNI
 *
 */

#include "dataloader.h"

#include <android/log.h>
#include <cstring>
#include <nntrainer_error.h>
#include <random>
#include <dirent.h>
#include "lodepng.h"


#define LOG_TAG "nntrainer"

#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace nntrainer::indoor
{

DataLoader::DataLoader (float *data_, int data_size_, int data_len_, int label_len_)
    : data (data_), data_size (data_size_), label_len (label_len_), data_len (data_len_)
{
  idxes = std::vector<unsigned int> (data_size);
  std::iota (idxes.begin (), idxes.end (), 0);
  std::shuffle (idxes.begin (), idxes.end (), rng);
  SampleSize = (data_len + 1);
  count = 0;
}

void
DataLoader::next (float **input, float **label, bool *last)
{
  auto fill_one_sample = [this](float *input_, float *label_, int index) {
    unsigned int lo = index * SampleSize;
    for (unsigned int i = 0; i < data_len; ++i) {
      (*input_) = (data)[lo + i];
      input_++;
    }
    lo = index * SampleSize + data_len;
    for(unsigned int i=0;i<label_len;++i){
      label_[i] = 0.0;
    }
    (label_)[(int)(data)[lo]] = 1.0;
  };
  
  // LOGI("**** %d --> %d: \n     ", count, idxes[count] );
  
  fill_one_sample (*input, *label, idxes[count]);

  // std::string o = "";
  // for(unsigned int i =0 ;i<data_len; ++i){
  //   o = o+" "+std::to_string((*input)[i]); 
  // }
  // o =o+":"+ std::to_string(data[idxes[count]*SampleSize+data_len]);  
  
  // o = o+"\n";
  // LOGI("       %s", o.c_str());
  
  count++;
  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
    std::shuffle (idxes.begin (), idxes.end (), rng);
  }
}

ImageDataLoader::ImageDataLoader (const char *directory_, int data_size_,
    int label_len_, int w, int h, bool is_train_)
    : data_size (data_size_), label_len (label_len_), width (w), height (h),
      is_train (is_train_)
{
  idxes = std::vector<unsigned int> (data_size);
  std::iota (idxes.begin (), idxes.end (), 0);
  std::shuffle (idxes.begin (), idxes.end (), rng);
  
  dir_path.assign(directory_);

  count = 0;
  total_image_num =0;

  struct dirent *de;
  std::string path;
  
  for(unsigned int i=0;i<label_len;++i){
    std::string l_dir = dir_path+std::to_string(i);
    DIR *dir = opendir(l_dir.c_str());
    unsigned long image_count =0;
    
    if(!dir){
      LOGE("No image data : %s", l_dir.c_str());
      throw::std::runtime_error("cannot find dir");
    }
    
    while((de=readdir(dir))){
      if(de->d_type != DT_DIR){
	++image_count;
	++total_image_num;
      }
    }
    num_images.push_back(image_count);
  }

  if(is_train && total_image_num <= label_len){
    LOGE("total number of image is less than number of label");
    throw::std::invalid_argument("total number of image is less than number of label");
  }

  unsigned int cnt=0;
  unsigned int idx =0;
  unsigned int *img_idx=new unsigned int[label_len];
  
  for(unsigned int i=0;i<label_len;++i)
    img_idx[i]=0;

  while(true){
    if(idx >= total_image_num)
      break;
    unsigned int class_id = cnt%label_len;
    if(num_images[class_id] > img_idx[class_id]){
      datas.push_back (std::make_pair (
          class_id, img_idx[class_id]));
      img_idx[class_id]++;
      idx ++;
    }
    cnt++;
  }

  delete[] img_idx;
}

void
read_png (const std::string path, float *input, uint &width, uint &height)
{

  std::vector<unsigned char> img;
  std::vector<unsigned char> png;
  int *data = new int[width * height];
  lodepng::decode (img, width, height, path, LCT_RGB);
  for (unsigned int i = 0; i < width * height; i++) {
    data[i] = img[3 * i] << 16 | img[3 * i + 1] << 8 | img[3 * i + 2];
  }
  uint8_t *d = (reinterpret_cast<uint8_t *> (data));
  for (unsigned int i = 0; i < width * height * 3; ++i) {
    input[i] = d[i];
  }
  
  delete[] data;
}

void
ImageDataLoader::next (float **input, float **label, bool *last)
{
  auto fill_one_sample
    = [this](float *input_, float *label_, int index) {
          unsigned int id;
          if (is_train) {
            id = index;
          } else {
            id = total_image_num - index;
          }

          unsigned int class_id = datas[id].first;
          unsigned int image_id = datas[id].second;

          std::string path;
	  path = dir_path;

          std::string file_name = path + std::to_string (class_id) + "/"
                                  + std::to_string (image_id) + ".png";
	  LOGI("filename : %s\n", file_name.c_str());
          read_png (file_name, input_, width, height);

	  // std::string pix="";
	  // for(unsigned int i=0;i<height;++i){
	  //   unsigned int id = i*width;
	  //   for(unsigned int j=0;j<width;++j){
	  //     pix = pix+" "+std::to_string(input_[id+j]);
	  //   }
	  //   pix = pix + "\n";
	  // }
	  // LOGI("%s\n", pix.c_str());

          std::memset (label_, 0.0, label_len * sizeof (float));

          label_[class_id] = 1.0;
        };

  // LOGI ("::::: %d / %d, %d : %lu \n", count, idxes.size(), data_size, datas.size() );

  // LOGI("**** %d --> %d:  %d %d\n", count, idxes[count], datas[idxes[count]].first, datas[idxes[count]].second);
  
  fill_one_sample (*input, *label, idxes[count]);

  count++;

  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
    std::shuffle (idxes.begin (), idxes.end (), rng);
  }
}
} // namespace nntrainer::indoor
