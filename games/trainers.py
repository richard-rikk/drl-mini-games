from tqdm import tqdm
import numpy as np
import model
import gym


MIN_EPSILON            = 0.001      #The epsilon can not be lower than this number
EPSILON_DECAY          = 0.99985    #Epsilon will be equal to epsilon * EPSILON_DECAY


class DQNTrainer():
    def __init__(self,inputDims:tuple, gameType:str) -> None:
        self.epsilon        = 1
        self.inputDims      = inputDims
        self.current_state  = None
        self.last_move      = None
        self.gametype       = gameType
        self.env            = gym.make(self.gametype)
        self.model          = model.DqnModel(inputDims, self.env.action_space.n)
        self.rewards        = []

    def train_model(self) -> None:
        print(f'Log file will be updated every {model.AGGREGATE_STATS_EVERY} steps!')

        self.env.reset()
        stepsSinceLastUpdate = 0
        self.current_state, reward, done, _ = self.env.step(self.env.action_space.sample()) # take a random action
        for i in tqdm(range(10_000)):
            #The agent steps this sets the last_move
            self.step()

            state, reward, done, _ = self.env.step(self.last_move) # take a random action
            self.model.update_replay_memory((self.current_state, self.last_move, reward, state, done))
            
            self.model.train(i)
            
            self.current_state = state
            self.rewards.append(reward)

            stepsSinceLastUpdate += 1
            if stepsSinceLastUpdate >= model.AGGREGATE_STATS_EVERY:
                stepsSinceLastUpdate = 0
                self.__update_model()

            #If finished a game start over.
            if done:
                self.env.reset()
            
            

        self.env.close()
        self.model.save()
    
    def step(self) -> None:
        #Make a decision based on epsilon
        if np.random.random() > self.epsilon:
            # Get the best action from our model
            if self.gametype == 'FrozenLake-v0':
                action = self.model.get_best_q_value(np.array([self.current_state]))
            else:
                action = self.model.get_best_q_value(self.current_state)
        else:
            # Get random action
            action = np.random.randint(0, self.env.action_space.n)
        
        #Epsilon must be decayed in each step
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)
        
        self.last_move = action
    
    #Have to call this function every if not match % AGGREGATE_STATS_EVERY or match == 1:
    #It will update the informations on the tensorboard and save the model if necessary    
    def __update_model(self) -> None:
        if len(self.rewards) == 0:
            return
        
        average_reward = np.mean(self.rewards)
        min_reward = min(self.rewards)
        max_reward = max(self.rewards)
        self.model.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=self.epsilon)

