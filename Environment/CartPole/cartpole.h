/**
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
