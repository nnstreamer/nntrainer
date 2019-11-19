#include "cartpole.h"
#include <iostream>
#include <stdio.h>

int main() {
  Env::State state;
  Env::CartPole cartpole;
  srand(time(NULL));
  std::vector<float> action;
  action=cartpole.sample();
  cartpole.init();
  for (int episode = 0; episode < 100; episode++) {
    cartpole.reset(&state);
    float total_reward = 0;
    int total_steps = 0;
    while (1) {
      action = cartpole.sample();
      cartpole.step(action,false, &state);
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
