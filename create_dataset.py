#!/usr/bin/env python3

from __future__ import division, print_function

import os
import numpy
import gym
import pickle 
import datetime

from gym_duckietown.envs import DuckietownEnv
import pyglet

def main(total_samples=1000000):

    env = gym.make('Duckie-SimpleSim-v0')
    obs = env.reset()

    env.render()
    train_set = []
    for itr in range(total_samples):
        action = env.action_space.sample()
        obs_old = obs.copy()
        obs, reward, done, info = env.step(action)
        train_set.append([obs_old.copy(), action.copy(), obs.copy(), 'Duckie-SimpleSim-v0'])
        if itr%100==0:
            print('Iteration: {}'.format(itr))
            print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

        if done:
            print('done!')
            obs = env.reset()
            env.render()     

    return train_set

if __name__ == "__main__":
    train_set = main()
    if not os.path.exists('/home/nithin/transfer_rl/.dataset'):
        os.makedirs('/home/nithin/transfer_rl/.dataset')
    time = datetime.datetime.now()
    timestamp = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute)
    pickle.dump(train_set, open( "{}_{}.p".format('Duckie-SimpleSim-v0', timestamp), "wb" ))