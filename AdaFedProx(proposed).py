#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# In[2]:



import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
# !pip install fl_implementation_utils

# from fl_implementation_utils import *
    


# In[3]:


def create_clients(data_dict):
    '''
    Return a dictionary with keys as client names and values as data and label lists.
    
    Args:
        data_dict: A dictionary where keys are client names, and values are tuples of data and labels.
                    For example, {'client_1': (data_1, labels_1), 'client_2': (data_2, labels_2), ...}
    
    Returns:
        A dictionary with keys as client names and values as tuples of data and label lists.
    '''
    return data_dict


# In[4]:


import tensorflow as tf

class SimpleMLP:
    @staticmethod
    def build():
        # Load the pre-trained ResNet101V2 model
        base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # Freeze all layers except the last two convolutional layers and the classification layer
#         for layer in base_model.layers[:-5]:
#             layer.trainable = False
        base_model.trainable=False
        # Create the transfer learning model by adding custom classification layers on top of the base model
        model2 = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # Adjust the number of output classes accordingly
        ])

        # Compile the model
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model2


# In[6]:



def test_model(X_test, Y_test,  model, comm_round):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
#     logits = model.predict(X_test)
    #print(logits)
    loss,accuracy=model.evaluate(X_test,Y_test)
#     loss = cce(Y_test, logits)
#     acc = accuracy_score( tf.argmax(Y_test, axis=1),tf.argmax(logits, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, accuracy, loss))
    return accuracy, loss


# In[7]:


def avg_weights(scaled_weight_list):
    '''Return the average of the listed scaled weights.'''
    num_clients = len(scaled_weight_list)
    
    if num_clients == 0:
        return None  # Handle the case where the list is empty
        
    avg_grad = list()
    
    # Get the sum of gradients across all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0) / num_clients
        avg_grad.append(layer_mean)
        
    return avg_grad


# In[8]:


import numpy as np
train1=np.load("train1.npy")
label1=np.load("label1.npy")
train2=np.load("train2.npy")
label2=np.load("label2.npy")
train3=np.load("train3.npy")
label3=np.load("label3.npy")
train4=np.load("train4.npy")
label4=np.load("label4.npy")
print("import sucessfull")


# In[9]:


# test1=np.load("test1.npy")
# one_hot_labels1=np.load("one_hot_labels1.npy")
test=np.load("test.npy")
one_hot_labels=np.load("one_hot_labels.npy")
# test2=np.load("test2.npy")
# one_hot_labels2=np.load("one_hot_labels2.npy")
# test3=np.load("test3.npy")
# one_hot_labels3=np.load("one_hot_labels3.npy")
print("import sucessfull")


# In[11]:


test=test/255
train1=train1/255
train2=train2/255
train3=train3/255
train4=train4/255
# test1=test1/255


# In[12]:


client_data1 = {
    'client1': (test, one_hot_labels),
    'client2': (test, one_hot_labels),
    'client3': (test, one_hot_labels),
    'client4': (test, one_hot_labels)
    
}
#create clients
test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    
}
#create clients
clients_batched = create_clients(client_data2)


# In[5]:


smlp_local = SimpleMLP()
global_model = smlp_local.build()


# In[5]:


global_model.summary()


# In[3]:


import random
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the action space range
action_low = 0.0
action_high = 1.0

# Define the DQN agent class with a neural network that outputs continuous actions
class DQNAgent:
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005

        self.model = self.build_model()
        self.memory_buffer = []  # Store experiences
        self.max_memory_buffer=2000

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='sigmoid'))  # Sigmoid for [0, 1] range
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def select_action(self, state):
#         if np.random.uniform(0,1) < self.exploration_proba:
#             return np.random.choice(range(1))
        action = self.model.predict(state)
        return action
   
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
#         print(self.exploration_proba)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state":state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    
       
    def train(self, batch_size=25):
#      if len(self.memory_buffer) < batch_size:
#         return
     import numpy as np
    # Randomly sample a batch of experiences from memory
     np.random.shuffle(self.memory_buffer)
     batch_sample = self.memory_buffer[0:batch_size]

    # Create lists to store current states and corresponding Q-target values
     current_states = []
     q_targets = []

     for experience in batch_sample:
        # We compute the Q-values of S_t
        q_current_state = self.model.predict(experience["current_state"])
        # We compute the Q-target using Bellman optimality equation
        q_target = experience["reward"]
        if not experience["done"]:
            q_target = q_target + self.gamma * np.max(self.model.predict(experience["next_state"]))
        # Append the current state and its corresponding Q-target
        current_states.append(experience["current_state"])
        q_targets.append(q_current_state)

    # Convert lists to NumPy arrays
     current_states = np.array(current_states)
     q_targets = np.array(q_targets)
#      print(current_states)
     

# Assuming your input data is stored in 'input_data'
     current_states = np.reshape(current_states, (current_states.shape[0], current_states.shape[2]))

# Now 'input_data' should have the shape (None, 20)
   

    # Train the model using the entire batch
     self.model.fit(current_states, q_targets, batch_size=batch_size, verbose=0)




# In[4]:


num_clients=4
state_size = 5 * num_clients
action_size = 1  # A single continuous action representing μ
# Define the action space range
action_low = 0.0
action_high = 1.0
fedprox= DQNAgent(state_size, action_size, action_low, action_high)


# In[ ]:





# In[ ]:





# In[5]:


import numpy as np

def calculate_Ek(Nc, Nk):
    Ek = 0.0
    for c in range(5):
        
            term = (Nc[c] / Nk) * np.log(Nc[0] / Nk)
            Ek += term
    return Ek

# Example usage:
def calculate_pk(Nk, N):
    
    return Nk/N



# In[9]:



Nc1 =[100,50,200,20,80]
Nc2 =[20,70,40,40,70]
Nc3 =[60,180,40,30,60]
Nc4 =[130,20,30,220,100]
Nk1=sum(Nc1)
Nk2=sum(Nc2)
Nk3=sum(Nc3)
Nk4=sum(Nc4)
# print(Nk2)
N=Nk1+Nk2+Nk3+Nk4


# In[10]:


E1=calculate_Ek(Nc1, Nk1)
E2=calculate_Ek(Nc2, Nk2)
E3=calculate_Ek(Nc3, Nk3)
E4=calculate_Ek(Nc4, Nk4)


# In[11]:


P1=calculate_pk(Nk1, N)
P2=calculate_pk(Nk2, N)
P3=calculate_pk(Nk3, N)
P4=calculate_pk(Nk4, N)


# In[ ]:





# In[12]:


states=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

states[0][0]=E1
states[0][1]=P1
states[0][2]=5
states[0][3]=0
states[0][4]=0

states[1][0]=E2
states[1][1]=P2
states[1][2]=5
states[1][3]=0
states[1][4]=0

states[2][0]=E3
states[2][1]=P3
states[2][2]=5
states[2][3]=0
states[2][4]=0

states[3][0]=E4
states[3][1]=P4
states[3][2]=5
states[3][3]=0
states[3][4]=0


# In[13]:


states=np.array(states)
print(states.shape)
states=states.reshape(-1,1)
print(states.shape)


# In[ ]:





# In[14]:


def calculate_fedprox_regularization(global_model, local_model, mu):
    regularization_term = 0.0
    global_weights = global_model.get_weights()
    local_weights = local_model.get_weights()

    for global_w, local_w in zip(global_weights, local_weights):
        regularization_term += 0.5 * mu * tf.reduce_sum(tf.square(global_w - local_w))

    return regularization_term


# In[15]:


states=np.array(states)
states=states.reshape(-1,1)
states.shape
states.reshape(1,20)
states.shape


# In[16]:


states=states.reshape(1,20)
print(states.shape)


# In[17]:


states=states.flatten()
print(states[3])


# In[18]:


# # states=np.array(states)
# # # print(states.shape)
# # states=states.flatten()
# # print(states.shape)
# states=states.reshape(-1,1)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# states = scaler.fit_transform(states)


# In[19]:


states.shape


# In[20]:


client_names = list(clients_batched.keys())


# In[ ]:


acc=[]
for i in range(60):
    acc.append(i)


# In[ ]:


# num_comm_rounds =60
# max_iteration_ep=10
# def training(states,acc):
#     for comm_round in range(num_comm_rounds):
#         batch_size=25
#         acc1=[]
#         for step in range(max_iteration_ep):
            
#         # Get the global model's weights - will serve as the initial weights for all local models
#               local_weight_list = []
#               client_updates = []
#               global_weights = global_model.get_weights()
            
              
#         # Calculate the current state based on the information you have
#               state = states  # Assuming state is a tuple of 5 values for each client

#         # Choose the action (mu value) for this round using the DQN agent
#               action = fedprox.select_action(np.array([state]))  # Convert state to a NumPy array
#               mu=action[0,0]
#               print(mu)
#               fedprox.model.save_weights("dqn_model_partial_weights.h5")


#               for i, client in enumerate(client_names):
#             # Simulate the communication round using the selected mu value (action) for each client

#             # Create a local model for the client
#                 smlp_local = SimpleMLP()
#                 local_model = smlp_local.build()

#             # Set the local model's weights to the global weights
#                 local_model.set_weights(global_weights)

#                 if client == 'client1':
#                     history = local_model.fit(
#                     np.array(clients_batched[client][0]),
#                     np.array(clients_batched[client][1]),
#                     validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                     epochs=2,
#                     batch_size=32,
#                     verbose=2
#                 )
#                 elif client == 'client2':
#                     history = local_model.fit(
#                     np.array(clients_batched[client][0]),
#                     np.array(clients_batched[client][1]),
#                     validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                     epochs=1,
#                     batch_size=32,
#                     verbose=2
#                 )

#                 elif client == 'client3':
#                      history=local_model.fit(
#               np.array(clients_batched[client][0]),
#             np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
#             np.array(test_batched[client][1])),
#             epochs=3,
#             batch_size=32,
#             verbose=2
#         ) 
#                 else:
#                     history=local_model.fit(
#             np.array(clients_batched[client][0]),
#             np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
#             np.array(test_batched[client][1])),
#             epochs=5,
#             batch_size=32,
#             verbose=2
#         ) 
            
#                 if i==0:
#                     state[3] = history.history['accuracy'][0]
#                     state[4] = history.history['loss'][0]
#                 elif i==1:
#                     state[8] = history.history['accuracy'][0]
#                     state[9] = history.history['loss'][0]
#                 elif i==2:
#                     state[13] = history.history['accuracy'][0]
#                     state[14] = history.history['loss'][0]
                
#                 else:
#                      state[18] = history.history['accuracy'][0]
#                      state[19] = history.history['loss'][0]    
                    
            

#             # Get the local model's weights after training
#                 local_weights = local_model.get_weights()
            

#             # Apply FedProx regularization
#                 fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)
#                 for i in range(len(local_weights)):
#                     local_weights[i] += mu * (global_weights[i] - local_weights[i])
                    
                    

#             # Append the local weights to the list
#                 # Calculate the difference between client model and server model
#                 client_update = []
#                 for server_layer, client_layer in zip(global_model.layers, local_model.layers):
#                    if isinstance(server_layer, tf.keras.layers.BatchNormalization):
#                 # Calculate the scaling factor based on the number of samples in each client
#                     scale_factor = lenlen(clients_batched[client][i]) / total_samples
#                 # Update the moving mean and variance of BatchNormalization layer
#                     updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
#                     updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance

#                 # Apply the updates to the server model's BatchNormalization layer
#                     server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
#                    else:
#                 # For other layers, update weights directly
#                     server_layer.set_weights(client_layer.get_weights())

#                 client_update.append(server_layer.get_weights())

#                 client_updates.append(client_update)
# #         print(client_updates)
# #         print(type(client_updates))
# #         client_updates=np.array(client_updates,dtype='object')

#     # Aggregate client updates to update the server model
#     # Manually calculate the mean of weights
#         aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]
# #                 local_weight_list.append(local_weights)

#             # Clear the session to free memory after each communication round
#             # K.clear_session()

#         # Calculate the average weights across all clients for each layer
# #               average_weights = avg_weights(local_weight_list)

#         # Update the global model with the average weights
# #               global_model.set_weights(average_weights)

#         # Test the global model and print out metrics after each communication round
#         for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
#             server_layer.set_weights(aggregated_weights)
#         global_acc, global_loss = test_model(test, one_hot_labels, global_model,step)

#         # Calculate the client-specific state for Q-learning
#         # state[client * 5 + 3] = history.history['accuracy'][0]
#         # state[client * 5 + 4] = history.history['loss'][0]

#         # Append the local weights to the list
# #         local_weight_list.append(local_weights)

#         # Calculate the average weights across all clients for each layer
# #         average_weights = avg_weights(local_weight_list)

#         # Update the global model with the average weights
# #         global_model.set_weights(average_weights)

#         # Test the global model and print out metrics after each communication round
# #               global_acc, global_loss = test_model(test1, one_hot_labels1, global_model, comm_round)

#         # After the communication round, calculate the reward
#         reward = global_acc  # Define your reward function based on global performance
#         acc1.append(reward)
#         acc[step]=max(acc1)

#         # Store the experience
#         new_state = state  # New state after taking the action
#         fedprox.store_experience(np.array([state]), action, reward, np.array([new_state]), done=False)
# #               if done:
# #                  agent.update_exploration_probability()
# #                  break
#         # Train the DQN agent
#         if step >= batch_size:
#                fedprox.train()

#         state = new_state  # Update the state for the next round
#     return acc            


# In[58]:


# def training(states, acc):
#     for comm_round in range(num_comm_rounds):
#         batch_size = 16
#         acc1 = []
        
#         for step in range(max_iteration_ep):
#             local_weight_list = []
#             client_updates = []
#             global_weights = global_model.get_weights()
#             state = states
            
#             action = fedprox.select_action(np.array([state]))
#             mu = action[0, 0]
#             print(mu)
#             fedprox.model.save_weights("dqn_model_partial_weights.h5")

#             for i, client in enumerate(client_names):
#                 smlp_local = SimpleMLP()
#                 local_model = smlp_local.build()
#                 local_model.set_weights(global_weights)

#                 if client == 'client1':
#                     epochs = 2
#                 elif client == 'client2':
#                     epochs = 1
#                 elif client == 'client3':
#                     epochs = 3
#                 else:
#                     epochs = 5

#                 history = local_model.fit(
#                     np.array(clients_batched[client][0]),
#                     np.array(clients_batched[client][1]),
#                     validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
#                     epochs=epochs,
#                     batch_size=32,
#                     verbose=2
#                 )

#                 # Update the state based on history
#                 # ...
#                 if i==0:
#                     state[3] = history.history['accuracy'][0]
#                     state[4] = history.history['loss'][0]
#                 elif i==1:
#                     state[8] = history.history['accuracy'][0]
#                     state[9] = history.history['loss'][0]
#                 elif i==2:
#                     state[13] = history.history['accuracy'][0]
#                     state[14] = history.history['loss'][0]
                
#                 else:
#                      state[18] = history.history['accuracy'][0]
#                      state[19] = history.history['loss'][0]
#                 local_weights = local_model.get_weights()
#                 fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)

#                 for i in range(len(local_weights)):
#                     local_weights[i] += mu * (global_weights[i] - local_weights[i])

#                 client_update = []

#                 for server_layer, client_layer in zip(global_model.layers, local_model.layers):
#                     if isinstance(server_layer, tf.keras.layers.BatchNormalization):
#                         scale_factor = len(clients_batched[client][i]) / total_samples
#                         updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
#                         updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance
#                         server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
#                     else:
#                         server_layer.set_weights(client_layer.get_weights())
#                     client_update.append(server_layer.get_weights())
#                 client_updates.append(client_update)

#             aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]

#             for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
#                 server_layer.set_weights(aggregated_weights)

#             global_acc, global_loss = test_model(test, one_hot_labels, global_model, step)

#             reward = global_acc
#             acc1.append(reward)
#             acc[step] = max(acc1)

#             new_state = state
#             fedprox.store_experience(np.array([state]), action, reward, np.array([new_state]), done=False)

#             if step >= batch_size:
#                 fedprox.train()

#             state = new_state

#     return acc


# In[28]:


import matplotlib.pyplot as plt


# In[93]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Assuming other necessary imports and variable definitions are present

num_comm_rounds = 50  # Replace with the actual value
max_iteration_ep = 10  # Replace with the actual value

# Initialize an empty list for storing accuracy values
accuracy = []

# Assuming other necessary imports and variable definitions are present

for comm_round in range(num_comm_rounds):
    batch_size = 16
    acc1 = []

    for step in range(max_iteration_ep):
        print(comm_round)
        local_weight_list = []
        client_updates = []
        global_weights = global_model.get_weights()
        state = states

        action = fedprox.select_action(np.array([state]))
        mu = action[0, 0]
        print(mu)
        fedprox.model.save_weights("dqn_model_full_weights_.h5")

        for i, client in enumerate(client_names):
            smlp_local = SimpleMLP()
            local_model = smlp_local.build()
            local_model.set_weights(global_weights)

            if client == 'client1':
                epochs = 5
            elif client == 'client2':
                epochs = 5
            elif client == 'client3':
                epochs = 5
            else:
                epochs = 5

            history = local_model.fit(
                np.array(clients_batched[client][0]),
                np.array(clients_batched[client][1]),
                validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
                epochs=epochs,
                batch_size=32,
                verbose=2
            )

            if i == 0:
                state[3] = history.history['accuracy'][0]
                state[4] = history.history['loss'][0]
            elif i == 1:
                state[8] = history.history['accuracy'][0]
                state[9] = history.history['loss'][0]
            elif i == 2:
                state[13] = history.history['accuracy'][0]
                state[14] = history.history['loss'][0]
            else:
                state[18] = history.history['accuracy'][0]
                state[19] = history.history['loss'][0]

            local_weights = local_model.get_weights()
            fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)

            for i in range(len(local_weights)):
                local_weights[i] += mu * (global_weights[i] - local_weights[i])

            client_update = []

            for server_layer, client_layer in zip(global_model.layers, local_model.layers):
                if isinstance(server_layer, tf.keras.layers.BatchNormalization):
                    scale_factor = len(clients_batched[client][i]) / total_samples
                    updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
                    updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance
                    server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
                else:
                    server_layer.set_weights(client_layer.get_weights())
                client_update.append(server_layer.get_weights())
            client_updates.append(client_update)

        aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]

        for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
            server_layer.set_weights(aggregated_weights)

        global_acc, global_loss = test_model(test, one_hot_labels, global_model, step)

        reward = global_acc
        acc1.append(reward)

        new_state = state
        fedprox.store_experience(np.array([state]), action, reward, np.array([new_state]), done=False)

        if step >= batch_size:
            fedprox.train()

        state = new_state

    # Get the maximum accuracy after 10 steps and store it in the accuracy list
    max_acc_10_steps = max(acc1)
    accuracy.append(max_acc_10_steps)
    plt.plot(accuracy)
    plt.show


# In[ ]:


# accuracy=training(states,acc)


# In[95]:


import matplotlib.pyplot as plt
plt.plot(accuracy)
# accuracy


# In[30]:


# new_agent= DQNAgent(state_size, action_size, action_low, action_high)
fedprox.model.load_weights("dqn_model_full_weights_.h5")

states=[[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

states[0][0]=E1
states[0][1]=P1
states[0][2]=2
states[0][3]=.97
states[0][4]=0.04

states[1][0]=E2
states[1][1]=P2
states[1][2]=1
states[1][3]=.94
states[1][4]=0.1

states[2][0]=E3
states[2][1]=P3
states[2][2]=1
states[2][3]=.95
states[2][4]=0.1

states[3][0]=E4
states[3][1]=P4
states[3][2]=1
states[3][3]=.96
states[3][4]=0.1
states=np.array(states)
states=states.flatten()
print(states.shape)

# Provide the current state as input to the DQN model
predicted_mu = fedprox.model.predict(np.array([states]))

# `predicted_mu` contains the predicted mu value for the given state
print("Predicted mu value:", predicted_mu)


# In[ ]:





# In[96]:


import pickle
# Save the DQN agent
with open("fedprox_new_full.pkl", "wb") as file:
    pickle.dump(fedprox, file)


# In[28]:


import pickle
with open("fedprox_new_full.pkl", "rb") as file:
    fedprox = pickle.load(file)


# In[31]:


acc1 = []  # Replace with the actual initialization

# Loop over communication rounds
# for comm_round in range(num_comm_rounds):
batch_size = 25

for step in range(50):
    local_weight_list = []
    client_updates = []
    global_weights = global_model.get_weights()
    state = states

    action = fedprox.select_action(np.array([state]))
    mu = action[0, 0]
    print(mu)
    fedprox.model.save_weights("dqn_model_partial_wmeightsssjsnse.h5")

    for i, client in enumerate(client_names):
        smlp_local = SimpleMLP()
        local_model = smlp_local.build()
        local_model.set_weights(global_weights)

        if client == 'client1':
            epochs = 5
        elif client == 'client2':
            epochs = 5
        elif client == 'client3':
            epochs = 5
        else:
            epochs = 5

        history = local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),
            validation_data=(np.array(test_batched[client][0]), np.array(test_batched[client][1])),
            epochs=epochs,
            batch_size=32,
            verbose=2
        )

        # Update the state based on history
        # ...
        if i == 0:
            state[3] = history.history['accuracy'][0]
            state[4] = history.history['loss'][0]
        elif i == 1:
            state[8] = history.history['accuracy'][0]
            state[9] = history.history['loss'][0]
        elif i == 2:
            state[13] = history.history['accuracy'][0]
            state[14] = history.history['loss'][0]

        else:
            state[18] = history.history['accuracy'][0]
            state[19] = history.history['loss'][0]
        local_weights = local_model.get_weights()
        fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)

        for i in range(len(local_weights)):
            local_weights[i] += mu * (global_weights[i] - local_weights[i])

        client_update = []

        for server_layer, client_layer in zip(global_model.layers, local_model.layers):
            if isinstance(server_layer, tf.keras.layers.BatchNormalization):
                scale_factor = len(clients_batched[client][i]) / total_samples
                updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
                updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance
                server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
            else:
                server_layer.set_weights(client_layer.get_weights())
            client_update.append(server_layer.get_weights())
        client_updates.append(client_update)

    aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object),
                                 axis=0) for i in range(len(client_updates[0]))]

    for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
        server_layer.set_weights(aggregated_weights)

    global_acc, global_loss = test_model(test, one_hot_labels, global_model, step)

    reward = global_acc
    acc1.append(reward)
    plt.plot(acc1)
    plt.show()
    # acc[step] = max(acc1)

    new_state = state
    fedprox.store_experience(np.array([state]), action, reward, np.array([new_state]), done=False)

    if step >= batch_size:
        fedprox.train()

    state = new_state


# In[30]:


# import matplotlib.pyplot as plt
# acc1 = []
# acc1 = training(states,acc1)
import matplotlib.pyplot as plt


# In[45]:


plt.plot(acc1)


# In[32]:


global_model.evaluate(test,one_hot_labels)


# In[99]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
acc4 = accuracy_score(y_true, y_pred)

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(y_true, y_pred)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Calculate Balanced Accuracy
bacc = balanced_accuracy_score(y_true, y_pred)

# Calculate F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass

# Calculate Precision for multiclass
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate Recall for multiclass
recall = recall_score(y_true, y_pred, average='weighted')



# # Calculate AUC (Area Under the Curve)
# # roc_auc = roc_auc_score(y_true, y_pred, average='weighted')

# # Create a confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# # Calculate Geometric Mean from the confusion matrix
# # tn, fp, fn, tp, = conf_matrix.ravel()
# # g_mean = (tp / (tp + fn)) * (tn / (tn + fp))**0.5

# # Print or use these metrics as needed
print("Accuracy:", acc4)
print("Cohen's Kappa:", kappa)
print("Matthews Correlation Coefficient:", mcc)
print("Balanced Accuracy:", bacc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
# print("AUC (Area Under the Curve):", roc_auc)
# print("Geometric Mean:", g_mean)
from sklearn.metrics import roc_auc_score

# Assuming you have true labels and predicted probabilities for each class
y_true = one_hot_labels # Replace with your true labels
y_prob = global_model.predict(test) # Replace with your predicted probabilities

# # Calculate AUC for multiclass classification
auc = roc_auc_score(y_true, y_prob, average='weighted')

# # Print or use the AUC value as needed
print("AUC (Area Under the Curve):", auc)
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming you have true labels and predicted labels for multiclass classification
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)  # Replace with your predicted labels

# Convert true and predicted labels to class labels (not one-hot encoded)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate G-Mean for each class
class_g_means = []
for class_label in range(5):  # Replace num_classes with the number of classes
    # Create a binary confusion matrix for the current class
    true_class = (y_true == class_label)
    pred_class = (y_pred == class_label)
    tn, fp, fn, tp = confusion_matrix(true_class, pred_class).ravel()

    # Calculate Sensitivity (True Positive Rate) and Specificity (True Negative Rate)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate G-Mean for the current class
    g_mean = np.sqrt(sensitivity * specificity)

    class_g_means.append(g_mean)

# Calculate the overall G-Mean (geometric mean of class G-Means)
overall_g_mean = np.prod(class_g_means) ** (1 / len(class_g_means))

# Print or use the overall G-Mean as needed
print("Overall G-Mean:", overall_g_mean)


# In[97]:


# acc 
acccc=np.array(accuracy)


# In[98]:


np.save("acc_adaptfedprox_full_newstraggler.npy",acccc)


# In[100]:


global_model.save("adaptfedprox(full_new).h5")


# In[ ]:




