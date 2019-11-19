#include "include/gym/gym.h"
#include "include/matrix.h"
#include "include/neuralnet.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <stdio.h>
#include <unistd.h>
#include <memory>

#ifdef USING_CUSTOM_ENV
#include "CartPole/cartpole.h"
#define STATE Env::State
#define ENV Env::CartPole
#else
#define STATE Gym::State
#define ENV Gym::Environment
#endif

#define MAX_EPISODS 50000
#define HIDDEN_LAYER_SIZE 50
#define RENDER true
#define REPLAY_MEMORY 50000
#define MINI_BATCH 30
#define DISCOUNT 0.9
#define TRAINING false
#define LEARNIG_RATE 0.001

typedef struct {
  STATE state;
  std::vector<float> action;
  float reward;
  STATE next_state;
  bool done;
} Experience;

static double RandomDouble(double min, double max) {
  double r = (double)rand() / (double)RAND_MAX;
  return min + r * (max - min);
}

static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

static std::vector<Experience> getMiniBatch(std::deque<Experience> Q) {
  int max = (MINI_BATCH > Q.size()) ? MINI_BATCH : Q.size();
  int min = (MINI_BATCH < Q.size()) ? MINI_BATCH : Q.size();

  bool duplicate[max];
  std::vector<int> mem;
  std::vector<Experience> in_Exp;
  int count = 0;

  for (int i = 0; i < max; i++)
    duplicate[i] = false;

  while (count < min) {
    int nomi = rangeRandom(0, Q.size() - 1);
    if (!duplicate[nomi]) {
      mem.push_back(nomi);
      duplicate[nomi] = true;
      count++;
    }
  }

  for (int i = 0; i < min; i++) {
    in_Exp.push_back(Q[mem[i]]);
  }

  return in_Exp;
}

static int argmax(std::vector<double> vec) {
  int ret = 0;
  double val = 0.0;
  for (unsigned int i = 0; i < vec.size(); i++) {
    if (val < vec[i]) {
      val = vec[i];
      ret = i;
    }
  }
  return ret;
}

static std::shared_ptr<ENV> init_environment(int &input_size,
                                               int &output_size) {

#ifdef USING_CUSTOM_ENV

  std::shared_ptr<ENV> env(new ENV);
  env->init();
  input_size = env->getInputSize();
  output_size = env->getOutputSize();
#else
  std::shared_ptr<Gym::Client> client;
  std::string env_id = "CartPole-v0";
  try {
    client = Gym::client_create("127.0.0.1", 5000);
  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return NULL;
  }

  std::shared_ptr<ENV> env = client->make(env_id);
  std::shared_ptr<Gym::Space> action_space = env->action_space();
  std::shared_ptr<Gym::Space> observation_space = env->observation_space();

  input_size = observation_space->sample().size();

  output_size = action_space->discreet_n;
#endif

  return env;
}

static bool is_file_exist(std::string filename) {
  std::ifstream infile(filename);
  return infile.good();
}

int main(int argc, char **argv) {
  std::string filepath = "debug.txt";
  std::string model_path = "model.bin";
  std::ofstream writeFile(filepath.data());

  writeFile.is_open();

  srand(time(NULL));
  std::deque<Experience> expQ;

  std::shared_ptr<ENV> env;

  int input_size, output_size;
  env = init_environment(input_size, output_size);
  printf("input_size %d, output_size %d\n", input_size, output_size);

  Network::NeuralNetwork mainNet;
  Network::NeuralNetwork targetNet;

  mainNet.init(input_size, HIDDEN_LAYER_SIZE, output_size, MINI_BATCH,
               LEARNIG_RATE, "tanh", true);

  mainNet.setOptimizer("adam", LEARNIG_RATE, 0.9, 0.999, 1e-8);

  targetNet.init(input_size, HIDDEN_LAYER_SIZE, output_size, MINI_BATCH,
                 LEARNIG_RATE, "tanh", true);

  if (is_file_exist(model_path)) {
    mainNet.readModel(model_path);
    std::cout << "read model file \n";
  }

  targetNet.copy(mainNet);

  for (int episode = 0; episode < MAX_EPISODS; episode++) {
    float epsilon = 1. / ((episode / 10) + 1);
    bool done = false;
    int step_count = 0;
    STATE s;
    STATE next_s;

    env->reset(&s);

    while (!done) {
      std::vector<float> action;
      double r = RandomDouble(0.0, 1.0);

      if (r < epsilon && TRAINING) {
#ifdef USING_CUSTOM_ENV
        action = env->sample();
#else
        std::shared_ptr<Gym::Space> action_space = env->action_space();
        action = action_space->sample();
#endif
        std::cout << "test result random action : " << action[0] << "\n";
      } else {
        std::vector<double> input(s.observation.begin(), s.observation.end());
        Matrix test = mainNet.forwarding(Matrix({input}));
        std::vector<double> temp = test.Mat2Vec();
        action.push_back(argmax(temp));

        std::cout << "test result : " << temp[0] << " : " << temp[1] << " ---> "
                  << argmax(temp) << " size of action : " << action.size()
                  << "\n";
      }

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
      if (done) {
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONE : Episode "
                  << episode << " Iteration : " << step_count << "\n";
        ex.reward = -100.0;
        if (!TRAINING)
          break;
      }

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

    if (episode % 10 == 1 && TRAINING) {
      for (int iter = 0; iter < 50; iter++) {
        std::vector<Experience> in_Exp = getMiniBatch(expQ);
        std::vector<std::vector<std::vector<double>>> inbatch;
        std::vector<std::vector<std::vector<double>>> next_inbatch;

        for (unsigned int i = 0; i < in_Exp.size(); i++) {
          STATE state = in_Exp[i].state;
          STATE next_state = in_Exp[i].next_state;
          std::vector<double> in(state.observation.begin(),
                                 state.observation.end());
          inbatch.push_back({in});

          std::vector<double> next_in(next_state.observation.begin(),
                                      next_state.observation.end());
          next_inbatch.push_back({next_in});
        }

        Matrix Q = mainNet.forwarding(Matrix(inbatch));

        Matrix NQ = targetNet.forwarding(Matrix(next_inbatch));
        std::vector<double> nqa = NQ.Mat2Vec();

        for (unsigned int i = 0; i < in_Exp.size(); i++) {
          if (in_Exp[i].done) {
            Q.setValue(i, 0, (int)in_Exp[i].action[0],
                       (double)in_Exp[i].reward);
          } else {
            double next = (nqa[i * NQ.getWidth()] > nqa[i * NQ.getWidth() + 1])
                              ? nqa[i * NQ.getWidth()]
                              : nqa[i * NQ.getWidth() + 1];
            Q.setValue(i, 0, (int)in_Exp[i].action[0],
                       (double)in_Exp[i].reward + DISCOUNT * next);
          }
        }
        mainNet.backwarding(Matrix(inbatch), Q, iter);
      }

      writeFile << "=== mainNet Loss : " << mainNet.getLoss()
                << " : targetNet Loss : " << targetNet.getLoss() << "\n";
      std::cout << "=== mainNet Loss : " << mainNet.getLoss()
                << " : targetNet Loss : " << targetNet.getLoss() << "\n";
      targetNet.copy(mainNet);
      mainNet.saveModel(model_path);
    }
  }

  writeFile.close();
  return 0;
}
