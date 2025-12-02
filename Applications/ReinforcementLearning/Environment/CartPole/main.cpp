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
 * @file	main.cpp
 * @date	04 December 2019
 * @brief	This is simple example to use Env CartPole-v0
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "cartpole.h"
#include <iostream>
#include <stdio.h>

int main() {
  Env::State state;
  Env::CartPole cartpole;
  srand(time(NULL));
  std::vector<float> action;
  action = cartpole.sample();
  cartpole.init();
  for (int episode = 0; episode < 100; episode++) {
    cartpole.reset(&state);
    float total_reward = 0;
    int total_steps = 0;
    while (1) {
      action = cartpole.sample();
      cartpole.step(action, false, &state);
      total_reward += state.reward;
      total_steps += 1;
      printf("action : %f --> state : %f %f %f %f\n", action[0],
             state.observation[0], state.observation[1], state.observation[2],
             state.observation[3]);
      if (state.done)
        break;
    }
    printf("episode %i finished in %i steps with reward %02f\n", episode,
           total_steps, total_reward);
  }
}
