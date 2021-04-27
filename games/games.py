import gym
import model
import numpy                    as np
import tensorflow               as tf
import tensorflow_probability   as tfp

from .constants             import CAR_GAME, LAKE_GAME
from tqdm                   import tqdm

#This class will always just load a model and play with the game
class Game:
    #modelType: 0 -> DQN 1 -> AC  2 -> Acv2
    #subdir only needed for Acv2 models.
    def __init__(self, gameType:str, modelID:int, modelType:int) -> None:
        self.env        = gym.make(gameType)
        self.gameType   = gameType
        self.curr_state = None

        if modelType == 1:
            self.model = model.AcModel(modelID=modelID, actionSpaceSize=self.env.action_space.n)
        elif modelType == 0:
            self.model = model.DqnModel(modelID=modelID)
            
    def play(self,steps=25) -> None:
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

    def evaulate(self, episodes=100) -> float:       
        episode_rewards = []
        
        for _ in tqdm(range(episodes)):
            rewards = []
            done    = False
            self.curr_state = self.env.reset()

            while not done:
                move = self.model.choose_action(np.array([self.curr_state]))
                self.curr_state, reward, done, _= self.env.step(move)
                rewards.append(reward)
            episode_rewards.append(sum(rewards))
            self.env.reset()

        return np.mean(episode_rewards)

class Gamev2:
    def __init__(self, gameType:str, modelID:int, subdir:str) -> None:
        self.env        = gym.make(gameType)
        self.gameType   = gameType
        self.modelID    = modelID
        self.model      = model.model_load(f'saved_models/{modelID}/{subdir}')
        self.curr_state = None
        self.last_move  = None

    def evaulate(self, episodes=100) -> float:       
        episode_rewards = []        
        for _ in tqdm(range(episodes)):
            rewards = []
            done    = False
            self.curr_state = self.env.reset()
            while not done:
                self.step()
                self.curr_state, reward, done, _= self.env.step(self.last_move)
                rewards.append(reward)
            episode_rewards.append(sum(rewards))
            self.env.reset()

        return np.mean(episode_rewards)

    
    def step(self) -> None:
        state = tf.convert_to_tensor([self.curr_state]) # Dont forget to add +1 dimension, this will be the batch dimension.
        if self.gameType == LAKE_GAME:
            state = tf.reshape(state, (1, 1, 1))
        
        _, probs             = self.model(state)
        action_probabilities = tfp.distributions.Categorical(probs=tf.squeeze(probs))
        action               = action_probabilities.sample()
        self.last_move       = int(action.numpy())      # Dont forget to remove the batch dimension




