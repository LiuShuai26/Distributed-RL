# Distributed Deep Reinforcement Learning

Distributed Deep Reinforcement Learning by Ray and TensorFlow.

This framework inspired by general-purpose RL training system **Rapid** from OpenAI.

Rapid framework:
![rapid-architecture@2x--1-](./tutorial/Pictures/rapid-architecture@2x--1-.png)
This framework:
![ddrlframework](./tutorial/Pictures/ddrlframework.jpg)

------

Tutorial (Chinese version)

- [Parallelize your algorithm by Ray (1)](tutorial/Parallelize%20your%20algorithm%20by%20Ray%20(1).md)
- [Parallelize your algorithm by Ray (2)](tutorial/Parallelize%20your%20algorithm%20by%20Ray%20(2).md)
- [Parallelize your algorithm by Ray (3)](tutorial/Parallelize%20your%20algorithm%20by%20Ray%20(3).md)

------

In short. This framework divides the reinforcement learning process into five parts:

- Replay buffer (option)
- Parameter server
- train (learn)
- rollout
- test

简单实验对比：

实验：LunarLanderContinuous-v2

算法：SAC

未调参，sac和dsac参数相同，dsac的worker数量：1。GPU：GTX1060

(dsac: distributed sac)

![dsac1w-sac](./tutorial/Pictures/dsac1w-sac.png)