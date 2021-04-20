from tqdm import tqdm
import numpy as np
import model
import gym

from .constants import CAR_GAME, LAKE_GAME, INPUT_DIMS


MIN_EPSILON    = 0.001       #The epsilon can not be lower than this number
EPSILON_DECAY  = 0.999985    #Epsilon will be equal to epsilon * EPSILON_DECAY

class Trainer():
    def __init__(self,gameType:str) -> None:
        self.inputDims        = INPUT_DIMS[gameType]
        self.current_state    = None
        self.last_move        = None
        self.gametype         = gameType
        self.env              = gym.make(self.gametype)
        self.rewards          = []
        self.episode_step_cnt = 0
    
    def show_log_info(self) -> None:
        print(f'Currently trained model ID: {self.model.modelID}')
        print(f'Current;y trained model location : {self.model.modelLoc}')
        print(f'Log file will be updated every {model.AGGREGATE_STATS_EVERY} steps!')
    
    #It will update the informations on the tensorboard and save the model if necessary    
    def update_model(self, epsilon=0) -> None:
        if len(self.rewards) == 0:
            return
        
        average_reward = np.mean(self.rewards)
        min_reward = min(self.rewards)
        max_reward = max(self.rewards)
        self.model.tensorboard.update_stats(reward_avg=average_reward,
                                            reward_min=min_reward, 
                                            reward_max=max_reward,
                                            step_cnt=self.episode_step_cnt, 
                                            epsilon=epsilon)
           


class DQNTrainer(Trainer):
    def __init__(self,gameType:str) -> None:
        super().__init__(gameType=gameType)
        self.epsilon  = 1
        self.model    = model.DqnModel(self.inputDims, self.env.action_space.n)
        

    def train_model(self, episode=10_000) -> None:
        self.show_log_info()

        for _ in tqdm(range(episode)):
            done                   = False
            self.rewards           = []
            self.current_state     = self.env.reset()
            self.episode_step_cnt  = 0
            while not done:
                self.step()

                state, reward, done, _ = self.env.step(self.last_move) # take a random action
                self.model.update_replay_memory((self.current_state, self.last_move, reward, state, done))

                self.model.train(isDone=done)
                self.rewards.append(reward)
                self.episode_step_cnt += 1
                self.current_state = state
            
            self.update_model(self.epsilon)

        self.env.close()
        self.model.save()

    
    def step(self) -> None:
        #Make a decision based on epsilon
        if np.random.random() > self.epsilon:
            # Get the best action from our model
            action = self.model.choose_action(np.array([self.current_state]))
        else:
            # Get random action
            action = np.random.randint(0, self.env.action_space.n)
        
        #Epsilon must be decayed in each step
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)
        
        self.last_move = action
    
class ACTrainer(Trainer):
    def __init__(self,gameType:str) -> None:
        super().__init__(gameType=gameType)
        self.model   = model.AcModel(self.inputDims, self.env.action_space.n)
    
    def train_model(self, episode=10_000) -> None:
        self.show_log_info()

        for _ in tqdm(range(episode)):
            done                   = False
            self.rewards           = []
            self.current_state     = self.env.reset()
            self.episode_step_cnt  = 0
            while not done:
                self.step()

                state, reward, done, _ = self.env.step(self.last_move) # take a random action

                self.model.train(self.current_state, self.last_move, reward, state, done)
                self.rewards.append(reward)
                self.episode_step_cnt += 1
                self.current_state = state
            
            self.update_model()
        
        self.env.close()
        self.model.save()
    
    def step(self) -> None:
        self.last_move = self.model.choose_action(np.array([self.current_state]))

