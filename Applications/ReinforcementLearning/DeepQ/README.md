# Reinforcement Learning with DeepQ

In this toy example, Reinforcement Learning with DeepQ Learning is implemented without any neural network framework like tensorflow or tensorflow-lite. In order to do that, some algorithms are implemented and  tested on Galaxy 9 & Ubuntu 16.04 PC. All codes are written in C++.

- Implement DeepQ Learning Algorithm
. Experience Replay
. Two Neural Network ( main & target Network ) to stabilization
- Fully Connected Layer Support
- Multi Layer Support ( Two FC Layer is used )
- Gradient Descent Optimizer Support
- ADAM Optimizer Support
- batch_size Support
- Softmax Support
- sigmoid/tanh Support

For the Environment,
- OpenAI/Gym Cartpole-v0 support
- Native Cartpole environment is implemented
. Maximum Iteration is 200.

The results is below.

<p align=center>
<img src =https://github.com/nnstreamer/nntrainer/blob/master/docs/images/de916e80-0b9f-11ea-9950-5c40d2bef8e4.gif width=300 >
<img src =https://github.com/nnstreamer/nntrainer/blob/master/docs/images/d2f17800-0b9e-11ea-8060-edfeacd6c71e.gif width=300 >
</p>
