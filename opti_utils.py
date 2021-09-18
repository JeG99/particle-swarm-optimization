import numpy as np
import math
from numpy.random import random

def ackley(x, a=20, b=0.2, c=2*math.pi):
    return -a * np.exp(-b*np.sum(np.square(x))/2) - np.exp(np.sum(np.cos(c*x))/2) + a + np.exp(1) 

def update_vel(v0, alpha, n, dim, best_g, particles_pos, particles_loc):
    v0 += alpha * random(size=(n, dim)) * best_g - particles_pos + alpha * random(size=(n, dim)) * particles_loc - particles_pos

