import os
import time
import random

from typing                      import Any

from tensorflow.python.keras.backend import update
from .modifiedTb                 import ModifiedTensorBoard
from tensorflow.keras.layers     import LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras            import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models     import Model, save_model, load_model
from collections                 import deque

import tensorflow as tf
import numpy      as np

#Hyper parameters for the DQN model
LEARNING_RATE          = 0.01       #The model learning rate (alfa)
REPLAY_MEMORY_SIZE     = 25_000     #How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000      #Minimum number of steps in a memory to start training
MINIBATCH_SIZE         = 64         #The number of steps which we give the agent in one fit
UPDATE_TARGET_EVERY    = 100        #The number of episodes after we train the target_model 
DISCOUNT               = 0.99       #How much should we consider the future rewards.
AGGREGATE_STATS_EVERY  = 1000       #After how many steps we want to update the tensorboard for the given model

class DqnModel():
    #If modelID given it loads a model. Used for testing.
    def __init__(self,  inputDims:tuple=(0,0), actionSpaceSize:int=0, modelID=None) -> None:
        if modelID:
            self.modelID  = modelID
            self.modelLoc = f'saved_models/{self.modelID}/'
            self.load()
            print(f'Loaded model successfully!')
            return

        self.modelID    = int(time.time())
        self.inDims     = inputDims
        self.actionCnt  = actionSpaceSize
        self.modelLoc   = f'saved_models/{self.modelID}/'

        #Main model : Gets trained every step if we have enough saved steps in memory
        self.main_model = self.__buildNetwork()

        #Target model : This will be used to predict every step
        self.target_model = self.__buildNetwork()
        self.target_model.set_weights(self.main_model.get_weights())

        #An array which stores the steps for training it elements : (current_state, action, reward, new_current_state, done) tupels
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #Custom tensorboard objects
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.modelID}-{int(time.time())}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        os.mkdir(self.modelLoc)

    def __buildNetwork(self) -> Any:
        input = Input(shape=self.inDims)

        layer = LSTM(1024, return_sequences=True)(input)
        layer = Dropout(0.1)(layer)

        layer = LSTM(512, return_sequences=True)(layer)
        layer = Dropout(0.1)(layer)

        layer = LSTM(258, return_sequences=False)(layer)
        layer = Dropout(0.1)(layer)

        layer = Dense(1024, activation=activations.relu)(layer)
        layer = Dropout(0.1)(layer)

        layer = Dense(512, activation=activations.relu)(layer)
        layer = Dropout(0.1)(layer)

        layer = Dense(258, activation=activations.relu)(layer)
        layer = Dropout(0.1)(layer)

        #The output layer we need linear because we are interested in all q values.
        layer = Dense(self.actionCnt, activation=activations.linear)(layer)

        model = Model(inputs=input, outputs=layer)
        model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])

        return model

    # We save the target model, because this is the best version
    def save(self)  -> None:
        save_model(self.target_model,  self.modelLoc+"model.h5",  save_format='h5')
    
    # We load the target model into the main_model so we can use get_best_q_value
    def load(self) -> None:
        self.main_model = load_model(self.modelLoc+'model.h5')

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition:tuple) -> None:
        self.replay_memory.append(transition)
    
    #Trains the main_model every step and updates the targetet_model if needed
    def train(self, isDone:bool) -> None:
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states        = [transition[0] for transition in minibatch]
        current_qs_list       = self.main_model.predict(np.array(current_states))

        # Get future states from minibatch, then query NN model for Q values
        new_current_states    = [transition[3] for transition in minibatch]
        future_qs_list        = self.target_model.predict(np.array(new_current_states))

        # Collecting the traning data
        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        
        # Fit on all samples as one batch, log only on terminal state
        self.main_model.fit(x=[np.array(X)], 
                            y=np.array(y), 
                            batch_size=MINIBATCH_SIZE, 
                            verbose=0, 
                            shuffle=False,
                            )


        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.main_model.get_weights())
            self.target_update_counter = 0
        
        if isDone:
            self.target_update_counter += 1
    
    # Queries main network for Q values given current observation space (environment state)
    # This will tell us which actions should we take (get_best_q_value)

    def choose_action(self, state:np.ndarray) -> int:
        return np.argmax(self.main_model.predict(state))
              
    
        



