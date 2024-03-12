#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
gpu=int(input("Which gpu number you would like to allocate:"))
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)


# #### import numpy as np
# import random
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle 
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# import time
# 
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D 
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense 
# from tensorflow.keras.optimizers import SGD 
# from tensorflow.keras import backend as K

# 
# import numpy as np
# import random
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelBinarizer 
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from sklearn.metrics import accuracy_score
# from tqdm import tqdm
# import time
# 
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras import backend as K
# # !pip install fl_implementation_utils
# 
# # from fl_implementation_utils import *
#     

# In[2]:


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


# In[11]:


import tensorflow as tf

class SimpleMLP:
    @staticmethod
    def build():
        # Load the pre-trained DenseNet201 model
        base_model = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        
        # Freeze all layers except the last two convolutional layers and the classification layer
        base_model.trainable = False

        # Create the transfer learning model by adding custom classification layers on top of the base model
        model2 = tf.keras.models.Sequential([
#             tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (28, 28))),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # Adjust the number of output classes accordingly
        ])

        # Compile the model
        model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model2


# In[4]:



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





# In[5]:


def calculate_fedprox_regularization(global_model, local_model, mu):
    regularization_term = 0.0
    global_weights = global_model.get_weights()
    local_weights = local_model.get_weights()

    for global_w, local_w in zip(global_weights, local_weights):
        regularization_term += 0.5 * mu * tf.reduce_sum(tf.square(global_w - local_w))

    return regularization_term


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[ ]:





# In[9]:


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


# In[12]:


smlp_local = SimpleMLP()
global_model = smlp_local.build()


# In[13]:


global_model.summary()


# In[14]:


num_comm_rounds=50
acc3=[]
mu=0.53713053

for comm_round in range(num_comm_rounds):
        # Get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # Initialize a list to collect local model weights after scaling
        local_weight_list = []

        # Randomly select clients to participate in this round
        client_names = list(clients_batched.keys())
#         np.random.shuffle(client_names)

        for client in client_names:
            # Create a local model for the client
            smlp_local = SimpleMLP()
            local_model = smlp_local.build()

            # Set the local model's weights to the global weights
            local_model.set_weights(global_weights)

            # Train the local model on the client's data for one epoch
              # Fit local model with client's data
#             history=local_model.fit(
#             np.array(clients_batched[client][0]),
#             np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
#             np.array(test_batched[client][1])),
#             epochs=2,
            
#             verbose=2
#         )
            
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
            
            elif client == 'client3':
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
               
        # Get the local model's weights after training
            local_weights = local_model.get_weights()

            # Apply FedProx regularization
            fedprox_regularization = calculate_fedprox_regularization(global_model, local_model, mu)
            for i in range(len(local_weights)):
                local_weights[i] += mu * (global_weights[i] - local_weights[i])

            # Append the local weights to the list
            local_weight_list.append(local_weights)
             # Clear the session to free memory after each communication round
#             K.clear_session()

        # Calculate the average weights across all clients for each layer
        average_weights = avg_weights(local_weight_list)

        # Update the global model with the average weights
        global_model.set_weights(average_weights)

        # Test the global model and print out metrics after each communication round
        global_acc, global_loss = test_model(test, one_hot_labels, global_model, comm_round)
        acc3.append(global_acc)


# In[17]:


import matplotlib.pyplot as plt
plt.plot(acc3)
# plt.ylim(.6,.8)
plt.grid(visible=True)


# In[19]:


global_model.evaluate(test,one_hot_labels)


# In[15]:


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



# Calculate AUC (Area Under the Curve)
# roc_auc = roc_auc_score(y_true, y_pred, average='weighted')

# Create a confusion matrix
# conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate Geometric Mean from the confusion matrix
# tn, fp, fn, tp, = conf_matrix.ravel()
# g_mean = (tp / (tp + fn)) * (tn / (tn + fp))**0.5

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


# In[16]:


acccc=np.array(acc3)
np.save("acc_ada_full_messidor",acccc)


# In[17]:


global_model.save("ada_full_messidor.h5")


# In[ ]:




