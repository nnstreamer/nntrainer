/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file	cartpole.h
 * @date	04 December 2019
 * @brief	This is environment class for cartpole example
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __CARTPOLE_H__
#define __CARTPOLE_H__

#include <cmath>
#include <iostream>
#include <vector>

/**
 * @Namespace   Namespace Env
 * @brief       Namespace Env
 */
namespace Env {
/**
 * @brief     State Data Type
 *            ovservation : state variables
 *            reward : reward
 *            done : boolean for end of episode
 */
typedef struct {
  std::vector<float> observation;
  float reward;
  bool done;
  std::string ginfo;
} State;

/**
 * @class   CartPole Class
 * @brief   CartPole-v0 example for Reinforcement Learning
 */
class CartPole {
 public:
  /**
   * @brief     Constructor of CartPole
   */
  CartPole(){};

  /**
   * @brief     Destructor of CartPole
   */
  ~CartPole(){};

  /**
   * @brief     Initialization fo CarPole variables
   *            Set hyper parameters & set observation zero
   */
  void init();

  /**
   * @brief     Run Env with action
   * @param[in] action input action
   * @param[in] rendering boolean variable to redering. (It is not used)
   * @param[out]s State Output calculated by Env
   */
  void step(const std::vector<float> &action, bool rendering, State *s);

  /**
   * @brief     reset Env
   * @param[out] initial_s copy inialize State from this->S
   */
  void reset(State *initial_s);

  /**
   * @brief     get InputSize : 4 (CartPole-v0 example)
   * @retval    inputsize
   */
  int getInputSize();

  /**
   * @brief     get OutputSize : 2 (CartPole-v0 example)
   * @retval    outputSize
   */
  int getOutputSize();

  /**
   * @brief     generate random action value
   * @retval    random action values as vector<float>
   */
  std::vector<float> sample();

 private:
  float gravity;
  float masscart;
  float masspole;
  float total_mass;
  float length;
  float polemass_length;
  float force_mag;
  float tau;
  std::string kinematics_integrator;
  float theta_threshold_radians;
  float x_threshold;
  int steps_beyond_done;
  int count;
  int action_dim;
  State S;
};
}

#endif
