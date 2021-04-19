import os
import time
import random
import numpy                    as np 
import tensorflow               as tf
import tensorflow.keras.backend as K

from typing                      import Tuple, Any
from .modifiedTb                 import ModifiedTensorBoard
from tensorflow.keras.layers     import LSTM, Dense, Dropout, Input
from tensorflow.keras            import activations
from tensorflow.keras.models     import Model, save_model, load_model
from tensorflow.keras.optimizers import Adam

#Hyper parameters for the AC model
LEARNING_RATE_ACTOR    = 0.0001     #The AC model A learning rate
LEARNING_RATE_CRITIC   = 0.0005     #The AC model C learning rate
DISCOUNT               = 0.99       #How much should we consider the future rewards.

#This is the policy network needed to connect the actor and the critic:
#y_true is an action that the actor took, and y_pred the probability of that action
def custom_logLikelihood(delta):
    def loss(y_true, y_pred):
        out     = K.clip(y_pred, 1e-8, 1-1e-8)         #Clip the prediction so it can not be 0 or 1.
        log_lik = y_true * K.log(out)
        return K.sum(-log_lik * delta)
    return loss

class AcModel():
    def __init__(self,  inputDims:tuple=(0,0), actionSpaceSize:int=0, modelID=None) -> None:
        if modelID:
           self.modelID     = modelID
           self.actionCnt   = actionSpaceSize
           self.modelLoc    = f'saved_models/{self.modelID}/'
           self.load()
           print(f'Loaded model successfully!')
           return

        self.modelID    = int(time.time())
        self.inDims     = inputDims
        self.actionCnt  = actionSpaceSize
        self.modelLoc   = f'saved_models/{self.modelID}/'

        self.actor, self.critic, self.policy, self.delta = self.__buildNetword()

        #Custom tensorboard objects
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.modelID}-{int(time.time())}")

        os.mkdir(self.modelLoc)
    
    def __buildNetword(self) -> Tuple :
        #Use this to calculate the loss function. This is can not be an Input layer because the loss function can not work with keras tensors.
        delta = tf.Variable([[0.]], trainable=False)

        input = Input(shape=self.inDims)

        layer = LSTM(128, return_sequences=True)(input)
        layer = Dropout(0.1)(layer)

        layer = LSTM(64, return_sequences=True)(layer)
        layer = Dropout(0.1)(layer)

        layer = LSTM(32, return_sequences=False)(layer)
        layer = Dropout(0.1)(layer)

        layer = Dense(32, activation=activations.relu)(layer)
        layer = Dropout(0.1)(layer)

        #------------------------------------------------These are the outputs of the  models------------------------------------------------
        #This is the output of the critic model. The input is one because we just want the value of 1 action, which was taken by the actor.
        values = Dense(1, activation=activations.linear)(layer)

        #This is the output of the actor model. We get the probabilities of each action, that is why softmax needed.
        probabilities = Dense(self.actionCnt, activation=activations.softmax)(layer)

        #------------------------------------------------These are the  models------------------------------------------------
        actor = Model(inputs=input, outputs=probabilities)
        actor.compile(loss=custom_logLikelihood(delta), optimizer=Adam(lr=LEARNING_RATE_ACTOR))
  
        critic = Model(inputs=input, outputs=values)
        critic.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE_CRITIC))

        #Dont compile because we dont backpropagate on this model.
        policy = Model(inputs=input, outputs=probabilities)

        return actor, critic, policy, delta
    
    # We save the target model, because this is the best version
    def save(self)  -> None:
        save_model(self.actor,  self.modelLoc+"actor.h5",   save_format='h5')
        save_model(self.actor,  self.modelLoc+"critic.h5",  save_format='h5')
        save_model(self.actor,  self.modelLoc+"policy.h5",  save_format='h5')
    
    def load(self) -> None:
        self.delta = tf.Variable([[0.]], trainable=False)
        self.actor  = load_model(self.modelLoc+"actor.h5", custom_objects={"loss":custom_logLikelihood(self.delta)})
        self.critic = load_model(self.modelLoc+"critic.h5",custom_objects={"loss":custom_logLikelihood(self.delta)})
        self.policy = load_model(self.modelLoc+"policy.h5",custom_objects={"loss":custom_logLikelihood(self.delta)}, compile=False)
    
    def choose_action(self, state:np.ndarray) -> int:
        probabilities = self.policy.predict(state)[0]
        action        = random.choices(np.arange(0,self.actionCnt), weights=probabilities, cum_weights=None, k=1)[0]
        return action
    
    #This is a temporal difference learning so we train on 1 example per fit.
    def train(self, prev_state:np.ndarray, action:int, reward:float, curr_state:np.ndarray, done:bool) -> None :
        prev_state = np.array([prev_state]) #We add +1 dimension. This will be the batch size
        curr_state = np.array([curr_state])

        critic_value_prev = self.critic.predict(prev_state)
        critic_value_curr = self.critic.predict(curr_state)

        target  = reward + DISCOUNT * critic_value_curr * (1-int(done))
        delta   = target - critic_value_prev
        
        #Set actions probability to 0 execpt the choosen action
        actions = np.zeros((1,self.actionCnt))
        actions[np.arange(1), action] = 1.0

        self.delta.assign(delta)
        self.actor.fit(x=prev_state,  y=actions, verbose=0)
        self.critic.fit(x=prev_state, y=target,  verbose=0)

       
