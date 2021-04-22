# Actor-Critic model implemented in tensorflow 2 style instead of tensorflow 1
# It should be the same as ac.py only using tf 2 convenctions
import os
import time
import numpy                    as np
import tensorflow               as tf
import tensorflow_probability   as tfp
from typing import Tuple

from tensorflow.python.keras import activations

from .modifiedTb                 import ModifiedTensorBoard
from tensorflow.keras.models     import Model, save_model, load_model
from tensorflow.keras.layers     import Dense, InputLayer
from tensorflow.keras            import activations
from copy                        import deepcopy

GAMMA = 0.99

class ACv2(Model):
    # 0 = CAR       1 = LAKE 
    def __init__(self,  gameType, inputDims:tuple=(0,0), actionSpaceSize:int=0, modelID=None) -> None:
        super(ACv2, self).__init__()
        
        self.modelID    = int(time.time())
        self.inDims     = deepcopy(inputDims)
        self.actionCnt  = actionSpaceSize
        self.modelLoc   = f'saved_models/{self.modelID}/'
        self.gameType   = gameType

        self.__buildNetwork()

        #Custom tensorboard objects
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.modelID}-{int(time.time())}")

        os.mkdir(self.modelLoc)
    
    def __buildNetwork(self) -> None:
        #self.layer_i = InputLayer(self.inDims)
        self.layer_1 = Dense(1024, activation=activations.relu)
        self.layer_2 = Dense(512, activation=activations.relu)

        #This is the value function approximation
        self.v  = Dense(1, activation=None)

        self.pi = Dense(self.actionCnt, activation=activations.softmax)

    def call(self, state):
        #value = self.layer_i(state)
        value = self.layer_1(state)
        value = self.layer_2(value)

        v  = self.v(value)
        pi = self.pi(value)

        return v, pi

def model_compile(model:ACv2) -> None:
    pass

def model_save(model: ACv2, idx:int) -> None:
    save_model(model,  model.modelLoc+f"acv2-{idx}.tf",   save_format='tf')

def model_load(location:str) -> ACv2:
    return load_model(location)

def model_train(model:ACv2, prev_state:np.ndarray, action:int, reward:float, curr_state:np.ndarray, done:bool) -> None :
    prev_state = tf.convert_to_tensor([prev_state], dtype=tf.float32)
    curr_state = tf.convert_to_tensor([curr_state], dtype=tf.float32)
    reward     = tf.convert_to_tensor([reward], dtype=tf.float32)
    
    if model.gameType:
        prev_state = tf.reshape(prev_state, (1, model.inDims[0], model.inDims[1]))
        curr_state = tf.reshape(curr_state, (1, model.inDims[0], model.inDims[1]))
  

    with tf.GradientTape() as tape:
        prev_state_value, probs = model(prev_state)
        curr_state_value, _     = model(curr_state)

        prev_state_value = tf.squeeze(prev_state_value)     #The values has to be a 1d array with 1 number
        prev_state_value = tf.squeeze(curr_state_value)

        action_probs = tfp.distributions.Categorical(probs=tf.squeeze(probs))
        log_prob     = action_probs.log_prob(action)

        delta       = reward * curr_state_value * (1 - int(done)) - prev_state_value
        actor_loss  = -log_prob * delta
        critic_loss = delta**2
        total_loss  = actor_loss + critic_loss
    
    gradient = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))