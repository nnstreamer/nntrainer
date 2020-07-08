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
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is DeepQ Reinforcement Learning Example
 *              Environment : CartPole-v0 ( from Native or Open AI / Gym )
 *              Support Experience Replay to remove data co-relation
 *              To maintain stability, two Neural Net are used ( mainNN,
 * targetNN )
 *
 *
 *                  +---------------------+              +----------+
 *                  |    Initialization   |------------->|          |
 *                  +---------------------+              |          |
 *                             |                         |          |
 *        +------->+-----------------------+             |          |
 *        |   +--->| Get Action from Q Net |             |          |
 *        |   |    +-----------------------+             |          |
 *        |   |                |                         |          |
 *        |   |     +---------------------+              |   Env    |
 *        |   |     |      Put Action     |------------->|          |
 *        |   |     +---------------------+              |          |
 *        |   |                |                         |          |
 *        |   |     +---------------------+              |          |
 *        |   |     |      Get State      |<-------------|          |
 *        |   |     +---------------------+              |          |
 *        |   |                |                         |          |
 *        |   |    +------------------------+            |          |
 *        |   |    | Set Penalty & Updaet Q |            |          |
 *        |   |    | from Target Network    |            |          |
 *        |   |    +------------------------+            |          |
 *        |   |                |                         |          |
 *        |   |    +-----------------------+             |          |
 *        |   +----| Put Experience Buffer |             |          |
 *        |        +-----------------------+             |          |
 *        |                    |                         |          |
 *        |        +------------------------+            |          |
 *        |        |  Training Q Network    |            |          |
 *        |        |     with minibatch     |            |          |
 *        |        +------------------------+            |          |
 *        |                    |                         |          |
 *        |        +------------------------+            |          |
 *        |        |    copy main Net to    |            |          |
 *        +--------|     Target Net         |            |          |
 *                 +------------------------+            +----------+
 *
 */

#include "neuralnet.h"
#include "tensor.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <stdio.h>
#include <unistd.h>

#ifdef USE_GYM
#include "include/gym/gym.h"
#define STATE Gym::State
#define ENV Gym::Environment
#define PTR boost::shared_ptr<ENV>
#else
#include "CartPole/cartpole.h"
#define STATE Env::State
#define ENV Env::CartPole
#define PTR std::shared_ptr<ENV>
#endif

/**
 * @brief     Maximum episodes to run
 */
#define MAX_EPISODES 50000

/**
 * @brief     boolean to reder (only works for openAI/Gym)
 */
#define RENDER true

/**
 * @brief     Max Number of data in Replay Queue
 */
#define REPLAY_MEMORY 50000

/**
 * @brief     minibach size
 */
#define MINI_BATCH 30

/**
 * @brief     discount factor
 */
#define DISCOUNT 0.9

/**
 * @brief     if true : update else : forward propagation
 */
#define TRAINING true

/**
 * @brief     Experience data Type to store experience buffer
 */
typedef struct {
  STATE state;
  std::vector<float> action;
  float reward;
  STATE next_state;
  bool done;
} Experience;

/**
 * @brief     Generate Random double value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static float RandomFloat(float Min, float Max) {
  float r = Min + static_cast<float>(rand()) /
                    (static_cast<float>(RAND_MAX) / (Max - Min));
  return r;
}

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int rangeRandom(int Min, int Max) {
  int n = Max - Min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return Min + x % n;
}

/**
 * @brief     Generate randomly selected Experience buffer from
 *            Experience Replay Queue which number is equal minibatch
 * @param[in] Q Experience Replay Queue
 * @retval    Experience vector
 */
static std::vector<Experience> getMiniBatch(std::deque<Experience> Q) {
  int Max = (MINI_BATCH > Q.size()) ? MINI_BATCH : Q.size();
  int Min = (MINI_BATCH < Q.size()) ? MINI_BATCH : Q.size();

  std::vector<bool> duplicate;
  std::vector<int> mem;
  std::vector<Experience> in_Exp;
  int count = 0;

  duplicate.resize(Max);

  for (int i = 0; i < Max; i++)
    duplicate[i] = false;

  while (count < Min) {
    int nomi = rangeRandom(0, Q.size() - 1);
    if (!duplicate[nomi]) {
      mem.push_back(nomi);
      duplicate[nomi] = true;
      count++;
    }
  }

  for (int i = 0; i < Min; i++) {
    in_Exp.push_back(Q[mem[i]]);
  }

  return in_Exp;
}

/**
 * @brief     Calculate argmax
 * @param[in] vec input to calculate argmax
 * @retval argmax
 */
static int argmax(std::vector<float> vec) {
  int ret = 0;
  float val = 0.0;
  for (unsigned int i = 0; i < vec.size(); i++) {
    if (val < vec[i]) {
      val = vec[i];
      ret = i;
    }
  }
  return ret;
}

/**
 * @brief     Create & initialize environment
 * @param[in] input_size State Size : 4 for cartpole-v0
 * @param[in] output_size Action Size : 2 for cartpole-v0
 * @retval Env object pointer
 */
static PTR init_environment(int &input_size, int &output_size) {
#ifdef USE_GYM
  boost::shared_ptr<Gym::Client> client;
  std::string env_id = "CartPole-v0";
  try {
    client = Gym::client_create("127.0.0.1", 5000);
  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return NULL;
  }

  boost::shared_ptr<ENV> env = client->make(env_id);
  boost::shared_ptr<Gym::Space> action_space = env->action_space();
  boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

  input_size = observation_space->sample().size();

  output_size = action_space->discreet_n;
#else
  std::shared_ptr<ENV> env(new ENV);
  env->init();
  input_size = env->getInputSize();
  output_size = env->getOutputSize();
#endif

  return env;
}

/**
 * @brief     Calculate DeepQ
 * @param[in]  arg 1 : configuration file path
 */
int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "./DeepQ Config.ini\n";
    exit(0);
  }
  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];

  std::string filepath = "debug.txt";
  std::ofstream writeFile(filepath.data());

  writeFile.is_open();

  srand(time(NULL));
  std::deque<Experience> expQ;

  PTR env;
  int status = 0;

  /**
   * @brief     Initialize Environment
   */
  int input_size, output_size;
  env = init_environment(input_size, output_size);
  printf("input_size %d, output_size %d\n", input_size, output_size);

  /**
   * @brief     Create mainNet & Target Net
   */
  nntrainer::NeuralNetwork mainNet(config);
  nntrainer::NeuralNetwork targetNet(config);

  /**
   * @brief     initialize mainNet & Target Net
   */
  mainNet.init();
  targetNet.init();

  /**
   * @brief     Read Model Data if any
   */
  mainNet.readModel();

  /**
   * @brief     Sync targetNet
   */
  targetNet.copy(mainNet);

  /**
   * @brief     Run Episode
   */
  for (int episode = 0; episode < MAX_EPISODES; episode++) {
    float epsilon = 1. / ((episode / 10) + 1);
    bool done = false;
    int step_count = 0;
    STATE s;
    STATE next_s;

    env->reset(&s);

    /**
     * @brief     Do until the end of episode
     */
    while (!done) {
      std::vector<float> action;
      float r = RandomFloat(0.0, 1.0);

      if (r < epsilon && TRAINING) {
#ifdef USE_GYM
        boost::shared_ptr<Gym::Space> action_space = env->action_space();
        action = action_space->sample();
#else
        action = env->sample();
#endif
        std::cout << "test result random action : " << action[0] << "\n";
      } else {
        std::vector<float> input(s.observation.begin(), s.observation.end());
        /**
         * @brief     get action with input State with mainNet
         */
        nntrainer::Tensor test =
          mainNet.forwarding(nntrainer::Tensor({input}), status);
        float *data = test.getData();
        unsigned int len = test.getDim().getDataLen();
        std::vector<float> temp(data, data + len);
        action.push_back(argmax(temp));

        std::cout << "qvalues : [";
        std::cout.width(10);
        std::cout << temp[0] << "][";
        std::cout.width(10);
        std::cout << temp[1] << "] : ACTION (argmax) = ";
        std::cout.width(3);
        std::cout << argmax(temp) << "\n";
      }

      /**
       * @brief     step Env with this action & save next State in next_s
       */
      env->step(action, RENDER, &next_s);
      Experience ex;
      ex.state = s;
      ex.action = action;
      ex.reward = next_s.reward;
      ex.next_state = next_s;
      ex.done = next_s.done;

      if (expQ.size() > REPLAY_MEMORY) {
        expQ.pop_front();
      }

      done = next_s.done;

      /**
       * @brief     Set Penalty or reward
       */
      if (done) {
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONE : Episode "
                  << episode << " Iteration : " << step_count << "\n";
        ex.reward = -100.0;
        if (!TRAINING)
          break;
      }

      /**
       * @brief     Save at the Experience Replay Buffer
       */
      expQ.push_back(ex);

      s = next_s;
      step_count++;

      if (step_count > 10000) {
        std::cout << "step_count is over 10000\n";
        break;
      }
    }
    if (step_count > 10000)
      break;

    if (!TRAINING && done)
      break;

    /**
     * @brief     Training after finishing 10 episodes
     */
    if (episode % 10 == 1 && TRAINING) {
      for (int iter = 0; iter < 50; iter++) {
        /**
         * @brief     Get Minibatch size of Experience
         */
        std::vector<Experience> in_Exp = getMiniBatch(expQ);
        std::vector<std::vector<std::vector<std::vector<float>>>> inbatch;
        std::vector<std::vector<std::vector<std::vector<float>>>> next_inbatch;

        /**
         * @brief     Generate Lable with next state
         */
        for (unsigned int i = 0; i < in_Exp.size(); i++) {
          STATE state = in_Exp[i].state;
          STATE next_state = in_Exp[i].next_state;
          std::vector<float> in(state.observation.begin(),
                                state.observation.end());
          inbatch.push_back({{in}});

          std::vector<float> next_in(next_state.observation.begin(),
                                     next_state.observation.end());
          next_inbatch.push_back({{next_in}});
        }

        /**
         * @brief     run forward propagation with mainNet
         */
        nntrainer::Tensor Q =
          mainNet.forwarding(nntrainer::Tensor(inbatch), status);

        /**
         * @brief     run forward propagation with targetNet
         */
        nntrainer::Tensor NQ =
          targetNet.forwarding(nntrainer::Tensor(next_inbatch), status);
        float *nqa = NQ.getData();

        /**
         * @brief     Update Q values & udpate mainNetwork
         */
        for (unsigned int i = 0; i < in_Exp.size(); i++) {
          if (in_Exp[i].done) {
            Q.setValue(i, 0, 0, (int)in_Exp[i].action[0],
                       (float)in_Exp[i].reward);
          } else {
            float next = (nqa[i * NQ.getWidth()] > nqa[i * NQ.getWidth() + 1])
                           ? nqa[i * NQ.getWidth()]
                           : nqa[i * NQ.getWidth() + 1];
            Q.setValue(i, 0, 0, (int)in_Exp[i].action[0],
                       (float)in_Exp[i].reward + DISCOUNT * next);
          }
        }
        mainNet.backwarding(nntrainer::Tensor(inbatch), Q, iter);
      }

      writeFile << "mainNet Loss : " << mainNet.getLoss()
                << " : targetNet Loss : " << targetNet.getLoss() << "\n";
      std::cout << "\n\n =================== TRAINIG & COPY NET "
                   "==================\n\n";
      std::cout << "mainNet Loss : ";
      std::cout.width(15);
      std::cout << mainNet.getLoss() << "\n targetNet Loss : ";
      std::cout.width(15);
      std::cout << targetNet.getLoss() << "\n\n";
      /**
       * @brief     copy targetNetwork
       */

      targetNet.copy(mainNet);
      mainNet.saveModel();
    }
  }

  /**
   * @brief     finalize networks
   */
  targetNet.finalize();
  mainNet.finalize();
  writeFile.close();
  return 0;
}
