import gym
import model
import numpy as np

from .constants             import CAR_GAME, LAKE_GAME
from numpy.core.fromnumeric import mean
from tqdm                   import tqdm

#This class will always just load a model and play with the game
class Game:
    #modelType: 0 -> DQN 1 -> AC
    def __init__(self, gameType:str, modelID:int, modelType:int) -> None:
        self.env        = gym.make(gameType)
        self.gameType   = gameType
        self.curr_state = None

        if modelType:
            self.model = model.AcModel(modelID=modelID, actionSpaceSize=self.env.action_space.n)
        else:
            self.model = model.DqnModel(modelID=modelID)
            
    def play(self,steps=25):
        self.env.reset()
        self.curr_state, _, _, _ = self.env.step(self.env.action_space.sample()) # take a random action
        for _ in range(steps):
            self.env.render()
            move = self.model.choose_action(np.array([self.curr_state]))
            self.curr_state, _, done, _= self.env.step(move)
            
            print(move)

            #If finished a game start over.
            if done:
                self.env.reset()
        self.env.close()

    def evaulate(self, episodes=100):
        self.env.reset()
        self.curr_state, _, _, _ = self.env.step(self.env.action_space.sample()) # take a random action
        done            = False
        episode_rewards = []
        
        for _ in tqdm(range(episodes)):
            rewards         = []

            while not done:
                move = self.model.choose_action(np.array([self.curr_state]))
                self.curr_state, reward, done, _= self.env.step(move)
                rewards.append(reward)
            episode_rewards.append(sum(rewards))
            self.env.reset()

        return mean(episode_rewards)




