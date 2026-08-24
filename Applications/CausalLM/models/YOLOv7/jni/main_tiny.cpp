// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main_tiny.cpp
 * @date   24 August 2026
 * @brief  YOLOv7-Tiny human-pet object detector (320x320, nc=4) inference on nntrainer.
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>
#include <util_func.h>
#include <app_context.h>

#include "yolov7_tiny_graph.h"

using ml::train::createModel;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

// Configuration
static std::string RES_DIR = ".";
static constexpr int NC = 4; // human, dog, cat, pet-etc
static constexpr int IMGSZ = 320;
static constexpr int NA = 3;
static constexpr int NO = NC + 5;

// Anchors
static constexpr float ANCHORS[3][3][2] = {
  {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}},      // Stride 8
  {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}},     // Stride 16
  {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}}  // Stride 32
};

struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int cls;
};

inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Failed to open binary file: " + path);
  }
  f.seekg(0, std::ios::end);
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> buf(size / sizeof(float));
  f.read(reinterpret_cast<char *>(buf.data()), size);
  return buf;
}

std::vector<Detection> decodeScale(const float *raw, int H, int W, float stride,
                                   int scale_idx, float conf_thres) {
  std::vector<Detection> dets;
  const int N = H * W;

  for (int a = 0; a < NA; ++a) {
    float aw = ANCHORS[scale_idx][a][0];
    float ah = ANCHORS[scale_idx][a][1];

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        int idx = y * W + x;
        const float *p = raw + (a * NO) * N + idx;

        float raw_obj = p[4 * N];
        float conf = sigmoid(raw_obj);
        if (conf < conf_thres)
          continue;

        float sx = sigmoid(p[0 * N]);
        float sy = sigmoid(p[1 * N]);
        float sw = sigmoid(p[2 * N]);
        float sh = sigmoid(p[3 * N]);

        float cx = ((sx * 2.0f) + ((float)x - 0.5f)) * stride;
        float cy = ((sy * 2.0f) + ((float)y - 0.5f)) * stride;
        float bw = (sw * sw) * 4.0f * aw;
        float bh = (sh * sh) * 4.0f * ah;

        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

        float best_score = 0.0f;
        int best_cls = -1;
        for (int c = 0; c < NC; ++c) {
          float score = sigmoid(p[(5 + c) * N]);
          if (score > best_score) {
            best_score = score;
            best_cls = c;
          }
        }

        float final_conf = conf * best_score;
        if (final_conf >= conf_thres) {
          dets.push_back({x1, y1, x2, y2, final_conf, best_cls});
        }
      }
    }
  }
  return dets;
}

std::vector<Detection> nms(std::vector<Detection> &candidates, float iou_thres, int max_det) {
  std::sort(candidates.begin(), candidates.end(),
            [](const Detection &a, const Detection &b) { return a.conf > b.conf; });

  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Detection> result;
  result.reserve(max_det);

  auto calc_iou = [](const Detection &a, const Detection &b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter;
    return union_area > 0.0f ? (inter / union_area) : 0.0f;
  };

  for (size_t i = 0; i < candidates.size() && result.size() < (size_t)max_det; ++i) {
    if (suppressed[i])
      continue;
    result.push_back(candidates[i]);
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && candidates[i].cls == candidates[j].cls) {
        if (calc_iou(candidates[i], candidates[j]) > iou_thres) {
          suppressed[j] = true;
        }
      }
    }
  }
  return result;
}

void printPeakRSS() {
  std::ifstream f("/proc/self/status");
  if (!f) return;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::cout << "[L1 Detector] Peak Memory RSS: " << line.substr(6) << std::endl;
      break;
    }
  }
}

int main(int argc, char **argv) {
  if (argc > 1) {
    RES_DIR = argv[1];
  }

  std::string input_path = RES_DIR + "/input_320.bin";
  if (argc > 2) {
    input_path = argv[2];
  }

  std::cout << "[L1 Detector] RES_DIR   = " << RES_DIR << std::endl;
  std::cout << "[L1 Detector] INPUT_BIN = " << input_path << std::endl;

  try {
    auto &app_ctx = nntrainer::AppContext::Global();
    (void)app_ctx;

    ModelHandle model = createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    bool preset_q = (std::getenv("YOLO_TENSOR_TYPE") != nullptr && 
                     std::string(std::getenv("YOLO_TENSOR_TYPE")) == "w8a32");

    auto x = Tensor(ml::train::TensorDim(1, 3, IMGSZ, IMGSZ,
                                         ml::train::TensorDim::Format::NCHW,
                                         ml::train::TensorDim::DataType::FP32),
                    "input0");

    auto outputs = yolov7_tiny::buildBackboneNeckHead(x, NC, preset_q);

    std::string weights_path = RES_DIR + "/yolov7_tiny.safetensors";
    if (preset_q) {
      // Check if pre-quantized weights exist
      std::string q8_path = RES_DIR + "/yolov7_tiny_q8.safetensors";
      std::ifstream f_q8(q8_path);
      if (f_q8.good()) {
        f_q8.close();
        weights_path = q8_path;
        std::cout << "[L1 Detector] Using offline pre-quantized weights: " << weights_path << std::endl;
      } else {
        std::cout << "[L1 Detector] Pre-quantized weights not found, fallback to on-the-fly quantization." << std::endl;
        setenv("NNTR_W8A8_FP32W", "1", 1);
      }
    }

    std::cout << "Compiling model..." << std::endl;
    if (model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE) != 0) {
      throw std::runtime_error("Model compilation failed!");
    }
    model->initialize();

    std::cout << "Loading weights: " << weights_path << std::endl;
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

    std::cout << "Running E2E inference..." << std::endl;
    std::vector<float> input = loadBin(input_path);
    std::vector<float *> in_ptr = {input.data()};

    // Warmup 5 runs to match PyTorch benchmark steady state
    for (int k = 0; k < 5; ++k) {
      auto outs = model->inference(1, in_ptr, std::vector<float *>());
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto outs = model->inference(1, in_ptr, std::vector<float *>());
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "[L1 Detector] Inference done in " << ms << " ms." << std::endl;
    printPeakRSS();

    // Print P3 sample output to verify exact correctness with PyTorch
    const float *p3 = outs[0];
    int N = 40 * 40;
    std::cout << "[L1 Detector] P3 scale output raw values at (a=0, y=0, x=0):" << std::endl;
    std::printf("  cx:   %.6g\n", p3[0 * N]);
    std::printf("  cy:   %.6g\n", p3[1 * N]);
    std::printf("  w:    %.6g\n", p3[2 * N]);
    std::printf("  h:    %.6g\n", p3[3 * N]);
    std::printf("  conf: %.6g\n", p3[4 * N]);
    std::printf("  c0:   %.6g\n", p3[5 * N]);
    std::printf("  c1:   %.6g\n", p3[6 * N]);
    std::printf("  c2:   %.6g\n", p3[7 * N]);
    std::printf("  c3:   %.6g\n", p3[8 * N]);

    // Decode scales
    std::vector<Detection> candidates;
    const int grids[3] = {40, 20, 10};
    const int strides[3] = {8, 16, 32};
    float conf_thres = std::getenv("YOLO_CONF") ? std::stof(std::getenv("YOLO_CONF")) : 0.25f;
    float iou_thres = std::getenv("YOLO_IOU") ? std::stof(std::getenv("YOLO_IOU")) : 0.45f;

    for (int i = 0; i < 3; ++i) {
      auto dets = decodeScale(outs[i], grids[i], grids[i], (float)strides[i], i, conf_thres);
      candidates.insert(candidates.end(), dets.begin(), dets.end());
    }

    auto dets = nms(candidates, iou_thres, 100);

    std::cout << "\n[";
    for (size_t i = 0; i < dets.size(); ++i) {
      const auto &d = dets[i];
      if (i) std::cout << ",";
      std::printf("\n  {\"x1\": %.6g, \"y1\": %.6g, \"x2\": %.6g, \"y2\": "
                  "%.6g, \"conf\": %.6g, \"cls\": %d}",
                  d.x1, d.y1, d.x2, d.y2, d.conf, d.cls);
    }
    std::cout << (dets.empty() ? "" : "\n") << "]" << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
