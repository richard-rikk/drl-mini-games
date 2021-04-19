import gym
import model
import time
import numpy as np

from .constants import CAR_GAME, LAKE_GAME

#This class will always just load a model and play with the game
class Game:
    def __init__(self, gameType:str, modelID:int) -> None:
        self.env        = gym.make(gameType)
        self.model      = model.DqnModel(modelID=modelID)
        self.env        = gym.make(gameType)
        self.gameType   = gameType
        self.curr_state = None
            
    def play(self,steps=25):
        self.env.reset()
        self.curr_state, _, _, _ = self.env.step(self.env.action_space.sample()) # take a random action
        for _ in range(steps):
            self.env.render()
            move = self.model.get_best_q_value(np.array([self.curr_state]))
            self.curr_state, _, done, _= self.env.step(move)
            
            print(move)

            #If finished a game start over.
            if done:
                self.env.reset()
        self.env.close()



