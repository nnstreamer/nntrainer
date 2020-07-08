/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.c
 * @date	04 May 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configureation.
 */

#include <nntrainer.h>

int main(int argc, char *argv[]) {
  int status = ML_ERROR_NONE;
  ml_nnmodel_h handle = NULL;
  const char *config_file = "./Tizen_CAPI_config.ini";
  status = ml_nnmodel_construct(&handle);
  if (status != ML_ERROR_NONE)
    return status;
  status = ml_nnmodel_compile_with_conf(config_file, handle);
  if (status != ML_ERROR_NONE)
    return status;
  status = ml_nnmodel_train_with_file(handle);
  if (status != ML_ERROR_NONE)
    return status;
  status = ml_nnmodel_destruct(handle);
  if (status != ML_ERROR_NONE)
    return status;
  return status;
}
