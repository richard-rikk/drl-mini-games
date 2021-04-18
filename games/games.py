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
        self.transformMapping = {CAR_GAME : self.convertInputCar, LAKE_GAME : self.convertInputLake}
            
    def play(self,setps=25):
        self.env.reset()
        state, _, _, _ = self.env.step(self.env.action_space.sample()) # take a random action
        for _ in range(setps):
            self.env.render()
            move = self.transformMapping[self.gameType](state)
            _, _, done, _= self.env.step(self.model.get_best_q_value(move))
            print(move)
            time.sleep(1)

            #If finished a game start over.
            if done:
                self.env.reset()
        self.env.close()

    #Converts the given input so it matches the requiremnts of the Lake type of networks
    def convertInputLake(self, state:int) -> np.ndarray:
        return np.array([state])
    
    def convertInputCar(self, state:np.ndarray) -> np.ndarray:
        return state



