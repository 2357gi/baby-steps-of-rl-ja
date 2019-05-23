#!/usr/bin/env python3
# coding: utf-8
"""
usage: python3 rlm1.py 
"""
# modules
import numpy as np
import matplotlib.pyplot as plt

# parameters
alpha = 0.2
reward_prob = 0.5
repetition = 100

# functions



# main
V_t = 0 # V_0 = 0
V_t_list = [ 0 ]
for i in range(repetition):
  # reward
  if np.random.uniform() < reward_prob:
    r_t = 1 # rewarded
  else:
    r_t = 0 # no reward

  # value update
  delta_t = r_t - V_t
  V_t += alpha*delta_t
  V_t_list.append( V_t )

# plot figures
plt.plot(range(len(V_t_list)), V_t_list, ".-")
plt.xlabel("Time t")
plt.ylabel("Value V_t")
plt.savefig("rlm1-plot.png")
plt.savefig("rlm1-plot.eps")
plt.clf()
