#!/usr/bin/env python3
# coding: utf-8
"""
usage: python3 rlm3.py 
"""
# modules
import numpy as np
import matplotlib.pyplot as plt

# parameters
alpha = 0.2 # 学習率
reward_prob = [0.5, 0.3]
repetition = 100



# functions
def softmax(Q_t):
  beta = 2.0 # 逆温度
  P_a = 1.0 / ( 1.0 + np.exp(-1.0*beta*(Q_t[0] - Q_t[1])) )
  return P_a


# main

# init. cond.
Q_t = [0, 0] # Q_a_0 = 0, Q_b_0 = 0
P_a = 0.5 # P_a=0.5 (P_b=1-P_a)
fig_data_list = [ [0], [0], [0.5], [0], [0]] # Q_a, Q_b, P_a (P_b=1-P_a), r_a, r_b


for i in range(repetition):
  # select a or b
  if np.random.uniform() < P_a:
    # a is selected
    selected = 0
  else:
    # b is selected
    selected = 1

  # reward
  if np.random.uniform() < reward_prob[selected]:
    r_t = 1 # rewarded
  else:
    #r_t = 0 # no reward
    r_t = -0.1 # negative reward

  # value update
  delta_t = r_t - Q_t[selected]
  Q_t[selected] += alpha*delta_t
  # 忘却率(1-alpha)有
  Q_t[1-selected] *= (1-alpha)

  # update selection probs. 
  P_a = softmax(Q_t)

  # save data for figures
  fig_data_list[0].append( Q_t[0] )
  fig_data_list[1].append( Q_t[1] )
  fig_data_list[2].append( P_a )
  fig_data_list[3+selected].append(r_t)
  fig_data_list[4-selected].append(0)

# plot figures
plt.subplot(5, 1, 1)
plt.plot(range(len(fig_data_list[0])), fig_data_list[0], "r.-")
#plt.xlabel("Time t")
plt.ylabel("Q_t(a)")

plt.subplot(5, 1, 2)
plt.plot(range(len(fig_data_list[1])), fig_data_list[1], "b.-")
#plt.xlabel("Time t")
plt.ylabel("Q_t(b)")

plt.subplot(5, 1, 3)
plt.plot(range(len(fig_data_list[2])), fig_data_list[2], "r.-")
#plt.xlabel("Time t")
plt.ylabel("P_a")

plt.subplot(5, 1, 4)
plt.plot(range(len(fig_data_list[3])), fig_data_list[3], "r.")
#plt.xlabel("Time t")
plt.ylabel("r_t(a)")

plt.subplot(5, 1, 5)
plt.plot(range(len(fig_data_list[4])), fig_data_list[4], "b.")
plt.xlabel("Time t")
plt.ylabel("r_t(b)")

plt.savefig("rlm3-plot.png")
plt.savefig("rlm3-plot.eps")
plt.clf()
