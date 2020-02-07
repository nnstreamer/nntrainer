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
 * @file	cartpole.cpp
 * @date	04 December 2019
 * @brief	This is environment class for cartpole example
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include "cartpole.h"
#include <stdlib.h>

#define M_PI 3.14159265358979323846

/**
 * @brief     Generate Random Double value between min to max
 * @retval    random value
 */
static double RandomDouble(double min, double max) { return min + ((double)rand() / (RAND_MAX / (max - min))); }

/**
 * @brief     Generate Random integer 0 or 1
 * @retval    random value
 */
static int random0to1() { return rand() % 2; }

namespace Env {

void CartPole::init() {
  gravity = 9.8;
  masscart = 1.0;
  masspole = 0.1;
  total_mass = masspole + masscart;
  length = 0.5;
  polemass_length = masspole * length;
  force_mag = 10.0;
  tau = 0.02;
  kinematics_integrator = "euler";
  theta_threshold_radians = 12 * 2 * M_PI / 360;
  x_threshold = 2.4;
  steps_beyond_done = -1;
  count = 0;
  action_dim = 2;
  for (int i = 0; i < 4; i++)
    S.observation.push_back(0.0);
}

void CartPole::step(const std::vector<float> &action, bool rendering, State *s) {
  double x = S.observation[0];
  double x_dot = S.observation[1];
  double theta = S.observation[2];
  double theta_dot = S.observation[3];
  double force = (action[0] == 1) ? force_mag : force_mag * -1.0;

  double costheta = cos(theta);
  double sintheta = sin(theta);
  double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
  double thetaacc =
      (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
  double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

  x = x + tau * x_dot;
  x_dot = x_dot + tau * xacc;
  theta = theta + tau * theta_dot;
  theta_dot = theta_dot + tau * thetaacc;

  S.observation[0] = x;
  S.observation[1] = x_dot;
  S.observation[2] = theta;
  S.observation[3] = theta_dot;
  s->observation.clear();
  s->observation.push_back(x);
  s->observation.push_back(x_dot);
  s->observation.push_back(theta);
  s->observation.push_back(theta_dot);

  S.done = (bool)(x < x_threshold * -1.0 || x > x_threshold || theta < theta_threshold_radians * -1.0 ||
                  theta > theta_threshold_radians);
  // theta > theta_threshold_radians || count >= 200);
  count++;
  s->done = S.done;

  if (!S.done) {
    S.reward = 1.0;
  } else if (steps_beyond_done == -1) {
    steps_beyond_done = 0;
    S.reward = 1.0;
  } else {
    if (steps_beyond_done == 0) {
      std::cout << "steps_beyond_done = 0 \n";
    }
    steps_beyond_done++;
    S.reward = 0.0;
  }
  s->reward = S.reward;
}

void CartPole::reset(State *initial_s) {
  initial_s->observation.clear();
  for (int i = 0; i < 4; i++) {
    S.observation[i] = RandomDouble(-0.05, 0.05);
    initial_s->observation.push_back(S.observation[i]);
  }
  steps_beyond_done = -1;
  S.reward = 0.0;
  initial_s->reward = 0.0;
  count = 0;
}

int CartPole::getInputSize() { return S.observation.size(); }

int CartPole::getOutputSize() { return action_dim; }

std::vector<float> CartPole::sample() {
  std::vector<float> action;
  action.push_back((float)random0to1());
  return action;
}
}
