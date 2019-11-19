#ifndef __CARTPOLE_H__
#define __CARTPOLE_H__

#include <cmath>
#include <iostream>
#include <vector>

namespace Env {

typedef struct {
  std::vector<float> observation;
  float reward;
  bool done;
  std::string ginfo;
} State;

class CartPole {
public:
  CartPole(){};
  ~CartPole(){};
  void init();
  void step(const std::vector<float> &action, bool rendering, State *s);
  void reset(State *initiali_s);
  int getInputSize();
  int getOutputSize();
  std::vector<float> sample();

private:
  double gravity;
  double masscart;
  double masspole;
  double total_mass;
  double length;
  double polemass_length;
  double force_mag;
  double tau;
  std::string kinematics_integrator;
  double theta_threshold_radians;
  double x_threshold;
  int steps_beyond_done;
  int count;
  int action_dim;
  State S;
};
}

#endif
