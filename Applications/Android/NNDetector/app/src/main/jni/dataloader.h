// SPDX-License-Identifier: Apache-2.0
/**
 * @file   dataloader.h
 * @date   24 Oct 2023
 * @brief  data handler for image
 * @author HS.Kim <hs0207.kim@samsung.com>
 * @bug    No known bugs
 */
#include <model.h>
#include <tensor_dim.h>

#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <android/log.h>
#define ANDROID_LOG_E(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, "nntrainer", fmt, ##__VA_ARGS__)
#define ANDROID_LOG_I(fmt, ...) \
  __android_log_print(ANDROID_LOG_INFO, "nntrainer", fmt, ##__VA_ARGS__)
#define ANDROID_LOG_D(fmt, ...) \
  __android_log_print(ANDROID_LOG_DEBUG, "nntrainer", fmt, ##__VA_ARGS__)

namespace nntrainer {
namespace simpleshot {

/**
 * @brief bounding box information
 *
 */
struct boundingBoxInfo {
  int micro_label;
  std::string file_name;
  int detected_macro_label;
  std::vector<int> bounding_box;
  float detection_score;
};

/**
 * @brief resize input
 *
 */
void resize_input(float *input, float *resized_input, uint &input_w,
                  uint &input_h, uint &new_dim);

/**
 * @brief normalize input
 *
 */
void normalize_input(float *input, uint &input_dim, std::vector<float> ch_mean,
                     std::vector<float> ch_std);

/**
 * @brief crop input
 *
 */
void crop_input(float *input, float *cropped_input, std::vector<int> xyxy,
                uint &old_dim);

/**
 * @brief read image
 *
 */
void read_image_from_path(const std::string path, float *input,
                          uint &new_img_dim);

/**
 * @brief argmax helper function
 *
 */
uint argmax(float *vec, unsigned int num_class);

/**
 * @brief user data object
 *
 */
class DirDataLoader {
public:
  /**
   * @brief Construct a new Directory Data Loader object
   *
   * @param directory directory path
   * @param label_len size of label data
   * @param image_dim size of square image
   * @param ch_mean normalizing vector for 3 channels
   * @param ch_std normalizing vector for 3 channels
   */
  DirDataLoader(const char *directory_, int label_len_, int new_img_dim_,
                std::vector<float> ch_mean_, std::vector<float> ch_std_);

  /**
   * @brief Destructor
   *
   */
  ~DirDataLoader() = default;

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

  /**
   * @brief set detector for the data loader
   */
  void runDetector(ml::train::Model *det_model, uint &det_input_img_dim_,
                   uint &det_output_dim, uint &det_anchor_num,
                   std::vector<uint> &det_labels, float &det_score_thr,
                   float &det_iou_thr, int &det_max_bb_num);

  /**
   * @brief get current file name
   */
  std::string getCurFileName();

  /**
   * @brief get class name
   */
  std::vector<std::string> getClassNames();

  /**
   * @brief get current boudning box information
   */
  boundingBoxInfo getCurrBBInfo();

  /**
   * @brief get bounding box list
   */
  std::vector<boundingBoxInfo> getBBList();

private:
  unsigned int data_size;
  std::vector<std::string> class_names;
  unsigned int label_len;
  unsigned int new_img_dim;
  std::vector<float> ch_mean;
  std::vector<float> ch_std;
  bool detector_set = false;
  unsigned int det_input_img_dim;
  std::vector<boundingBoxInfo> bounding_box_list;
  unsigned int count;
  std::vector<std::pair<unsigned int, std::string>> data_list;
  std::string cur_file_name;
  boundingBoxInfo curr_bb_info;
  std::string dir_path;
};

} // namespace simpleshot
} // namespace nntrainer
