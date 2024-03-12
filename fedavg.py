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


# def create_clients(data_list, label_list, num_clients=3, initial='clients'):
#     ''' return: a dictionary with keys clients' names and value as 
#                 data shards - tuple of datas and label lists.
#         args: 
#             data_list: a list of numpy arrays of training data
#             label_list:a list of binarized labels for each data
#             num_client: number of fedrated members (clients)
#             initials: the clients'name prefix, e.g, clients_1 
            
#     '''

#     #create a list of client names
#     client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

#     #randomize the data
#     data = list(zip(data_list, label_list))
#     random.shuffle(data)

#     #shard data and place at each client
#     size = len(data)//num_clients
#     shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

#     #number of clients must equal number of shards
#     assert(len(shards) == len(client_names))

#     return {client_names[i] : shards[i] for i in range(len(client_names))}

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


get_ipython().run_line_magic('run', 'data_augmentation_(1).ipynb')


# In[ ]:


# def batch_data(data_shard, bs=32):
#     '''Takes in a clients data shard and create a tfds object off it
#     args:
#         shard: a data, label constituting a client's data shard
#         bs:batch size
#     return:
#         tfds object'''
#     #seperate shard into data and labels lists
#     data, label = zip(*data_shard)
#     dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
#     return dataset.shuffle(len(label)).batch(bs)


# In[ ]:





# In[ ]:


import tensorflow as tf

class SimpleMLP:
    @staticmethod
    def build():
        # Load the pre-trained ResNet101V2 model
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
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


# In[ ]:



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


# In[ ]:





# In[ ]:


# def weight_scalling_factor(clients_trn_data, client_name):
#     client_names = list(clients_trn_data.keys())
#     #get the bs
#     bs = list(clients_trn_data[client_name])[0][0].shape[0]
#     #first calculate the total training data points across clinets
#     global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
#     # get the total number of data points held by a client
#     local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
#     return local_count/global_count


# def scale_model_weights(weight, scalar):
#     '''function for scaling a models weights'''
#     weight_final = []
#     steps = len(weight)
#     for i in range(steps):
#         weight_final.append(scalar * weight[i])
#     return weight_final

import tensorflow as tf

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


# In[ ]:


# import numpy as np
# train1=np.load("train1.npy")
# label1=np.load("label1.npy")
# train2=np.load("train2.npy")
# label2=np.load("label2.npy")
# train3=np.load("train3.npy")
# label3=np.load("label3.npy")
# train4=np.load("train4.npy")
# label4=np.load("label4.npy")
# print("import sucessfull")


# In[ ]:


# print(train1.shape)
# print(label1.shape)
# print(train2.shape)
# print(label2.shape)
# print(train3.shape)
# print(label3.shape)
# print(train4.shape)
# print(label4.shape)


# In[ ]:


import numpy as np
# test1=np.load("test1.npy")
# one_hot_labels1=np.load("one_hot_labels1.npy")
test=np.load("test.npy")
one_hot_labels=np.load("one_hot_labels.npy")
# test2=np.load("test2.npy")
# one_hot_labels2=np.load("one_hot_labels2.npy")
# test3=np.load("test3.npy")
# one_hot_labels3=np.load("one_hot_labels3.npy")
print("import sucessfull")


# In[ ]:





# In[ ]:


test=test/255


# In[ ]:


# test=test/255
# train1=train1/255
# train2=train2/255
# train3=train3/255
# train4=train4/255


# In[ ]:


# test1=test1/255
# test2=test2/255
# test3=test3/255


# In[ ]:


print(test.shape)
print(one_hot_labels.shape)
print(test1.shape)
print(one_hot_labels1.shape)


# In[ ]:


# print(train1.shape)
# print(label1.shape)
# print(train2.shape)
# print(label2.shape)
# print(train3.shape)
# print(label3.shape)


# In[ ]:


client_data1 = {
    'client1': (test, one_hot_labels),
    'client2': (test, one_hot_labels),
    'client3': (test, one_hot_labels),
    'client4': (test, one_hot_labels)
    
}
#create clients
test_batched = create_clients(client_data1)


# In[ ]:





# In[ ]:


client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    
}
#create clients
clients_batched = create_clients(client_data2)


# In[ ]:





# In[ ]:


# initialize global model
# print(data_list.shape,labels)
smlp_global = SimpleMLP()
global_model = smlp_global.build()
        


# In[ ]:


global_model.summary()


# In[ ]:


# len(clients_batched[client][1])
# global_model.get_weights()
client_names = list(clients_batched.keys())
client_names


# In[ ]:





# In[ ]:





# In[ ]:


comms_round = 50  # Number of global epochs
acc3 = []
train_acc_clients = [[], [], [],[]]  # List of lists for training accuracy for each client
val_acc_clients = [[], [], [],[]]    # List of lists for validation accuracy for each client


for comm_round in range(comms_round):

    # Get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # Initial list to collect local model weights after scaling
    local_weight_list = []

#     Randomize client data - using keys
    client_names = list(clients_batched.keys())
#     random.shuffle(client_names)

    for i, client in enumerate(client_names):

        smlp_global = SimpleMLP()
        local_model = smlp_global.build()

        local_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
#         num_local_epochs = client_epochs[client]
#         for _ in range(num_local_epochs):
#                 history = local_model.fit(
#                     np.array(clients_batched[client][0]),
#                     np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),np.array(clients_batched[client][1])),
#                     epochs=1,
#                     verbose=2
#                 )

        # Fit local model with client's data
        if client == 'client1':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=32,
            verbose=2
        )
        elif client == 'client2':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=32,
            verbose=2
        )
            
        else:
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=1,
            batch_size=32,
            verbose=2
        ) 
#         else:
#             history=local_model.fit(
#             np.array(clients_batched[client][0]),
#             np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
#             np.array(test_batched[client][1])),
#             epochs=1,
#             batch_size=32,
#             verbose=2
#         ) 
            
            
        
        # Store the training accuracy and validation accuracy for this client in this communication round
        train_acc_clients[i].append(history.history['accuracy'][0])
        val_acc_clients[i].append(history.history['val_accuracy'][0])

        # Get the scaled model weights and add to the list
        weights = local_model.get_weights()
        local_weight_list.append(weights)

        # Clear the session to free memory after each communication round
        K.clear_session()

    # Calculate the average weights across all clients for each layer
    average_weights = avg_weights(local_weight_list)

    # Update the global model with the average weights
    global_model.set_weights(average_weights)

    # Test the global model and print out metrics after each communications round
#     for (X_test, Y_test) in test_batched:
    global_acc, global_loss = test_model(test, one_hot_labels, global_model, comm_round)
    acc3.append(global_acc)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(acc3,color='red')
plt.title("FL vs communication rounds")
plt.grid(visible=True)
plt.legend(loc='right')
# plt.ylim(.6,.8)
plt.xlabel("FL rounds/Data sharing epochs")
plt.ylabel("Test accuracy")


# In[ ]:


import tensorflow as tf

# Replace 'path/to/fedavg(75%new straggler).h5' with the actual path to your H5 file
model_path = 'fedavg(75%new straggler).h5'

# Load the model
global_model = tf.keras.models.load_model(model_path)

# Display model summary
# fedavg_model.summary()


# In[ ]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
acc = accuracy_score(y_true, y_pred)


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
print("Accuracy:", acc)
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
y_true = one_hot_labels  # Replace with your true labels
y_prob = global_model.predict(test)  # Replace with your predicted probabilities

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


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[0],label='train')
plt.plot(range(1,len(acc3)+1),val_acc_clients[0],color='red',label='val')
# plt.ylim(0.95,.999)


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[1])
plt.plot(range(1,len(acc3)+1),val_acc_clients[1],color='red')
plt.ylim(0.90,.999)


# In[ ]:


plt.plot(range(1,len(acc3)+1),train_acc_clients[0])
plt.plot(range(1,len(acc3)+1),val_acc_clients[0],color='red')
plt.ylim(0.5,.999)


# In[ ]:


acccc=np.array(acc3)


# In[ ]:


np.save("acc_fedavg_MobileNetV2_balanced",acccc)


# In[ ]:


global_model.save("acc_fedavg_MobileNetV2_balanced.h5")


# In[ ]:


global_model.evaluate(test1,one_hot_labels1)


# In[ ]:


# !pip list


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, MaxPooling2D, AveragePooling2D, concatenate, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D
def Conv_2D_Block(x, model_width, kernel, strides=(1, 1), padding="same"):
    # 2D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv2D(model_width, kernel, strides=strides, padding=padding, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x
def Inception_Module_A(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch5x5 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch5x5 = Conv_2D_Block(branch5x5, filterB2_2, (5, 5))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 3))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1, name='Inception_Block_A'+str(i))

    return out

def Inception_Module_B(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch7x7 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (1, 7))
    branch7x7 = Conv_2D_Block(branch7x7, filterB2_2, (7, 1))

    branch7x7dbl = Conv_2D_Block(inputs, filterB3_1, 1)
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_2, (7, 1))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (1, 7))
    branch7x7dbl = Conv_2D_Block(branch7x7dbl, filterB3_3, (7, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1, name='Inception_Block_B'+str(i))

    return out
def Inception_Module_C(inputs, filterB1_1, filterB2_1, filterB2_2, filterB3_1, filterB3_2, filterB3_3, filterB4_1, i):
    # Inception Block i
    branch1x1 = Conv_2D_Block(inputs, filterB1_1, (1, 1))

    branch3x3 = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3_2 = Conv_2D_Block(branch3x3, filterB2_2, (1, 3))
    branch3x3_3 = Conv_2D_Block(branch3x3, filterB2_2, (3, 1))

    branch3x3dbl = Conv_2D_Block(inputs, filterB3_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (1, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB3_2, (3, 1))
    branch3x3dbl_2 = Conv_2D_Block(branch3x3dbl, filterB3_3, (1, 3))
    branch3x3dbl_3 = Conv_2D_Block(branch3x3dbl, filterB3_3, (3, 1))

    branch_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = Conv_2D_Block(branch_pool, filterB4_1, (1, 1))

    out = tf.keras.layers.concatenate([branch1x1, branch3x3_2, branch3x3_3, branch3x3dbl_2, branch3x3dbl_3, branch_pool], axis=-1, name='Inception_Block_C'+str(i))

    return out

def Reduction_Block_A(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block A (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (3, 3))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out

def Reduction_Block_B(inputs, filterB1_1, filterB1_2, filterB2_1, filterB2_2, filterB2_3, i):
    # Reduction Block B (i)
    branch3x3 = Conv_2D_Block(inputs, filterB1_1, (1, 1))
    branch3x3 = Conv_2D_Block(branch3x3, filterB1_2, (3, 3), strides=(2, 2))

    branch3x3dbl = Conv_2D_Block(inputs, filterB2_1, (1, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (1, 7))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_2, (7, 1))
    branch3x3dbl = Conv_2D_Block(branch3x3dbl, filterB2_3, (3, 3), strides=(2, 2))

    branch_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    out = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1, name='Reduction_Block_'+str(i))

    return out
def SE_Block(inputs, num_filters, ratio):
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    excitation = tf.keras.layers.Dense(units=num_filters/ratio)(squeeze)
    excitation = tf.keras.layers.Activation('relu')(excitation)
    excitation = tf.keras.layers.Dense(units=num_filters)(excitation)
    excitation = tf.keras.layers.Activation('sigmoid')(excitation)
    excitation = tf.keras.layers.Reshape([1, 1, num_filters])(excitation)

    scale = inputs * excitation

    return scale


# In[ ]:


class SEInception:
    def __init__(self, length, width, num_channel, num_filters, ratio=4, problem_type='Classification',
                 output_nums=5, pooling='avg', dropout_rate=False, auxilliary_outputs=False):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Model
        # kernel_size: Kernel or Filter Size of the Input Convolutional Layer
        # num_channel: Number of Channels of the Input Predictor Signals
        # problem_type: Regression or Classification
        # output_nums: Number of Output Classes in Classification mode and output features in Regression mode
        # pooling: Choose either 'max' for MaxPooling or 'avg' for Averagepooling
        # dropout_rate: If turned on, some layers will be dropped out randomly based on the selected proportion
        # auxilliary_outputs: Two extra Auxullary outputs for the Inception models, acting like Deep Supervision
        self.length = length
        self.width = width
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.ratio = ratio
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.auxilliary_outputs = auxilliary_outputs

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs


    def SEInception_v4(self):
        inputs = tf.keras.Input((self.length, self.width, self.num_channel))  # The input tensor
        # Stem
        x = Conv_2D_Block(inputs, 32, 3, strides=2, padding='valid')
        x = Conv_2D_Block(x, 32, 3, padding='valid')
        x = Conv_2D_Block(x, 64, 3)

        branch1 = Conv_2D_Block(x, 96, 3, strides=2, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 64, 1)
        branch1 = Conv_2D_Block(branch1, 96, 3, padding='valid')
        branch2 = Conv_2D_Block(x, 64, 1)
        branch2 = Conv_2D_Block(branch2, 64, 7)
        branch2 = Conv_2D_Block(branch2, 96, 3, padding='valid')
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        branch1 = Conv_2D_Block(x, 192, 3, padding='valid')
        branch2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = tf.keras.layers.concatenate([branch1, branch2], axis=-1)

        # 4x Inception-A Blocks - 35 x 256
        for i in range(4):
            x = Inception_Module_A(x, 96, 64, 96, 64, 96, 96, 96, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_0 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 0
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 96, 1)
            aux_output_0 = self.MLP(aux_conv)

        x = Reduction_Block_A(x, 64, 384, 192, 224, 256, 1)  # Reduction Block 1: 17 x 768

        # 7x Inception-B Blocks - 17 x 768
        for i in range(7):
            x = Inception_Module_B(x, 384, 192, 256, 192, 224, 256, 128, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)

        aux_output_1 = []
        if self.auxilliary_outputs:
            # Auxilliary Output 1
            aux_pool = tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(x)
            aux_conv = Conv_2D_Block(aux_pool, 128, 1)
            aux_output_1 = self.MLP(aux_conv)

        x = Reduction_Block_B(x, 192, 192, 256, 320, 320, 2)  # Reduction Block 2: 8 x 1280

        # 3x Inception-C Blocks: 8 x 2048
        for i in range(3):
            x = Inception_Module_C(x, 256, 384, 512, 384, 512, 512, 256, i)
            x = SE_Block(x, int(np.shape(x)[-1]), self.ratio)
        x = tf.expand_dims(x,axis = 1)
        x = ConvLSTM2D(filters=512, kernel_size=(1,1),padding = "same")(x)
        x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001)(x)
        x = Activation('relu')(x)

        # Final Dense MLP Layer for the outputs
        final_output = self.MLP(x)
        # Create model.
        model = tf.keras.Model(inputs, final_output, name='Inception_v4')
        if self.auxilliary_outputs:
            model = tf.keras.layers.Model(inputs, outputs=[final_output, aux_output_0, aux_output_1], name='Inception_v4')

        return model


# In[ ]:


model1=SEInception(128,128,3,64)
global_model=model1.SEInception_v4()


# In[ ]:


global_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )


# In[ ]:




