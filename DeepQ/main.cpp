#include "include/gym/gym.h"
#include "include/matrix.h"
#include "include/neuralnet.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <stdio.h>

#define MAX_EPISODS 10000
#define HIDDEN_LAYER_SIZE 40
#define RENDER true
#define REPLAY_MEMORY 50000
#define MINI_BATCH 10
#define DISCOUNT 0.9
#define TRAINING true

typedef struct {
  Gym::State state;
  std::vector<float> action;
  float reward;
  Gym::State next_state;
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
  // std::cout << "MINI_BATCH : "<< MINI_BATCH <<" q.size : " <<Q.size()<<"\n";

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

#if 0
static void run_single_environment(const boost::shared_ptr<Gym::Client> &client,
                                   const std::string &env_id,
                                   int episodes_to_run) {
  boost::shared_ptr<Gym::Environment> env = client->make(env_id);
  boost::shared_ptr<Gym::Space> action_space = env->action_space();
  boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

  for (int e = 0; e < episodes_to_run; ++e) {
    printf("%s episode %i...\n", env_id.c_str(), e);
    Gym::State s;
    env->reset(&s);
    float total_reward = 0;
    int total_steps = 0;
    while (1) {
      std::vector<float> action = action_space->sample();
      env->step(action, true, &s);
      assert(s.observation.size() == observation_space->sample().size());
      total_reward += s.reward;
      total_steps += 1;
      if (s.done)
        break;
    }
    printf("%s episode %i finished in %i steps with reward %0.2f\n",
           env_id.c_str(), e, total_steps, total_reward);
  }
}
#endif

static boost::shared_ptr<Gym::Environment>
init_environment(const boost::shared_ptr<Gym::Client> &client,
                 const std::string &env_id, int &input_size, int &output_size) {
  boost::shared_ptr<Gym::Environment> env = client->make(env_id);
  boost::shared_ptr<Gym::Space> action_space = env->action_space();
  boost::shared_ptr<Gym::Space> observation_space = env->observation_space();

  input_size = observation_space->sample().size();

  output_size = action_space->discreet_n;

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

  boost::shared_ptr<Gym::Client> client;
  boost::shared_ptr<Gym::Environment> env;
  boost::shared_ptr<Gym::Space> action_space;

  int input_size, output_size;
  try {
    client = Gym::client_create("127.0.0.1", 5000);
  } catch (const std::exception &e) {
    fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }

  env = init_environment(client, "CartPole-v0", input_size, output_size);
  printf("input_size %d, output_size %d\n", input_size, output_size);

  Network::NeuralNetwork mainNet;
  Network::NeuralNetwork targetNet;

  mainNet.init(input_size, HIDDEN_LAYER_SIZE, output_size, 0.9);
  targetNet.init(input_size, HIDDEN_LAYER_SIZE, output_size, 0.9);

  if (is_file_exist(model_path)) {
    mainNet.readModel(model_path);
    std::cout << "read model file \n";
  }

  targetNet.copy(mainNet);
  // writeFile << "init loss " << mainNet.getLoss() << "\n";

  for (int episode = 0; episode < MAX_EPISODS; episode++) {
    float epsilon = 1. / ((episode / 10) + 1);
    bool done = false;
    int step_count = 0;
    Gym::State s;
    Gym::State next_s;

    env->reset(&s);

    while (!done) {
      std::vector<float> action;
      double r = RandomDouble(0.0, 1.0);

      if (r < epsilon) {
        action_space = env->action_space();
        action = action_space->sample();
        // std::cout <<" epsilon : r "<< epsilon << " : "<<r  <<"\n";
        std::cout << "test result random action : " << action[0] << "\n";
      } else {
        std::vector<double> input(s.observation.begin(), s.observation.end());
        Matrix test = mainNet.forwarding(input);
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
                  << episode << " \n";
        ex.reward = -100;
      }

      expQ.push_back(ex);

      s = next_s;
      step_count++;

      if (step_count > 10000)
        break;
    }
    if (step_count > 10000)
      break;

    if (episode % 10 == 1 && TRAINING) {
      for (int iter = 0; iter < 50; iter++) {
        std::vector<Experience> in_Exp = getMiniBatch(expQ);
        for (unsigned int i = 0; i < in_Exp.size(); i++) {
          Gym::State state = in_Exp[i].state;
          Gym::State next_state = in_Exp[i].next_state;

          std::vector<double> in(state.observation.begin(),
                                 state.observation.end());
          Matrix Q = mainNet.forwarding(in);
          std::vector<double> qa = Q.Mat2Vec();
          std::vector<double> next_in(next_state.observation.begin(),
                                      next_state.observation.end());
          Matrix NQ = targetNet.forwarding(next_in);
          std::vector<double> nqa = NQ.Mat2Vec();
          double next = (nqa[0] > nqa[1]) ? nqa[0] : nqa[1];

          if (in_Exp[i].done) {
            qa[in_Exp[i].action[0]] = (double)in_Exp[i].reward;
          } else {
            qa[in_Exp[i].action[0]] =
                (double)(in_Exp[i].reward + DISCOUNT * next);
          }

          std::vector<double> _in(qa.begin(), qa.end());
          mainNet.backwarding(_in);
        }
      }

      writeFile << "===================== Loss : " << mainNet.getLoss()
                << " mainNet\n";
      std::cout << "\n\n===================== Loss : " << mainNet.getLoss()
                << " mainNet\n";

      targetNet.copy(mainNet);
      mainNet.saveModel(model_path);
    }
  }
  writeFile.close();
  return 0;
}
