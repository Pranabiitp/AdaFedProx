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

# In[2]:


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


# In[3]:


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


# In[4]:


print(train1.shape+train2.shape+train3.shape+train4.shape)


# In[5]:


# test1=np.load("test1.npy")
# one_hot_labels1=np.load("one_hot_labels1.npy")
test=np.load("test.npy")
one_hot_labels=np.load("one_hot_labels.npy")
# test2=np.load("test2.npy")
# one_hot_labels2=np.load("one_hot_labels2.npy")
# test3=np.load("test3.npy")
# one_hot_labels3=np.load("one_hot_labels3.npy")
print("import sucessfull")


# In[6]:


test=test/255
train1=train1/255
train2=train2/255
train3=train3/255
train4=train4/255
# test1=test1/255


# In[7]:


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


# In[8]:


client_data1 = {
    'client1': (test, one_hot_labels),
    'client2': (test, one_hot_labels),
    'client3': (test, one_hot_labels),
    'client4': (test, one_hot_labels)
    
}
#create clients
# test_batched = create_clients(client_data1)
client_data2 = {
    'client1': (train1, label1),
    'client2': (train2, label2),
    'client3': (train3, label3),
    'client4': (train4, label4),
    
}


# In[9]:


# train3.shape


# In[10]:


test_batched = create_clients(client_data1)
clients_batched = create_clients(client_data2)


# In[11]:


len(clients_batched['client1'][0])


# In[12]:


import tensorflow as tf
# import tensorflow_federated as tff
import numpy as np

# Define a simple model
def create_model():
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


# In[13]:


global_model=create_model()
global_model.summary()


# In[14]:


client_names = list(clients_batched.keys())
total_samples=1560


# In[15]:


# Federated training loop
acc=[]
num_rounds=1000
for epoch in range(num_rounds):
    # List to store client updates
    client_updates = []
    global_weights = global_model.get_weights()

    # Train each client model
    for i,client in enumerate(client_names):
        local_model = create_model()
        local_model.set_weights(global_weights)

        if client == 'client1':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=5,
            batch_size=32,
            verbose=2
        )
        elif client == 'client2':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=5,
            batch_size=32,
            verbose=2
        )
            
        elif client == 'client3':
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=5,
            batch_size=32,
            verbose=2
        ) 
        else:
            history=local_model.fit(
            np.array(clients_batched[client][0]),
            np.array(clients_batched[client][1]),validation_data=(np.array(test_batched[client][0]),
            np.array(test_batched[client][1])),
            epochs=5,
            batch_size=32,
            verbose=2
        ) 
            

        
        # Calculate the difference between client model and server model
        client_update = []
        for server_layer, client_layer in zip(global_model.layers, local_model.layers):
            if isinstance(server_layer, tf.keras.layers.BatchNormalization):
                # Calculate the scaling factor based on the number of samples in each client
                scale_factor = lenlen(clients_batched[client][i]) / total_samples
                # Update the moving mean and variance of BatchNormalization layer
                updated_mean = (1 - scale_factor) * server_layer.moving_mean + scale_factor * client_layer.moving_mean
                updated_variance = (1 - scale_factor) * server_layer.moving_variance + scale_factor * client_layer.moving_variance

                # Apply the updates to the server model's BatchNormalization layer
                server_layer.set_weights([updated_mean, updated_variance, server_layer.gamma, server_layer.beta])
            else:
                # For other layers, update weights directly
                server_layer.set_weights(client_layer.get_weights())

            client_update.append(server_layer.get_weights())

        client_updates.append(client_update)
#         print(client_updates)
#         print(type(client_updates))
#         client_updates=np.array(client_updates,dtype='object')

    # Aggregate client updates to update the server model
    # Manually calculate the mean of weights
    aggregated_update = [np.mean(np.array([client_update[i] for client_update in client_updates], dtype=object), axis=0) for i in range(len(client_updates[0]))]




    # Apply the aggregated update to the server model
    for server_layer, aggregated_weights in zip(global_model.layers, aggregated_update):
        server_layer.set_weights(aggregated_weights)
        
    global_acc, global_loss = test_model(test, one_hot_labels, global_model,epoch)
    acc.append(global_acc)    


# In[16]:


import matplotlib.pyplot as plt
plt.plot(acc,color='red')
plt.title("FL vs communication rounds")
plt.grid(visible=True)
plt.legend(loc='right')
# plt.ylim(.6,.8)
plt.xlabel("FL rounds/Data sharing epochs")
plt.ylabel("Test accuracy")


# In[17]:


from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, f1_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

# Assuming you have predictions and true labels
y_true = one_hot_labels  # Replace with your true labels
y_pred = global_model.predict(test)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Calculate Accuracy
accc3 = accuracy_score(y_true, y_pred)

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
print("Accuracy:", accc3)
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


# In[18]:


global_model.save("fedbn_full_new.h5")


# In[19]:


acccc=np.array(acc)
np.save("acc_fedbn_full_new",acccc)


# In[20]:


acc


# In[21]:


import tensorflow as tf

# Assuming input shape is (height, width, channels)
input_shape = (224, 224, 3)  # Adjust according to your actual input shape

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),  # Add input shape
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),  # Add Batch Normalization layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # Add another Batch Normalization layer
    tf.keras.layers.Dense(5, activation='softmax')  # Adjust the number of output classes accordingly
])

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[22]:


model2.summary()


# In[23]:


# Print indices of Batch Normalization layers
for i, layer in enumerate(model2.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print(f"Batch Normalization layer at index {i}")


# In[24]:


import tensorflow as tf
index_of_bn_layer=1
# Assuming bn_layer is your Batch Normalization layer
bn_layer = model2.layers[index_of_bn_layer]  # Replace with the actual index of your BN layer

# Get the statistics
mean = bn_layer.moving_mean.numpy()
variance = bn_layer.moving_variance.numpy()
scale = bn_layer.gamma.numpy()
offset = bn_layer.beta.numpy()

# Calculate standard deviation from variance
std_dev = tf.sqrt(variance)

# Shifted mean and scaled standard deviation
shifted_mean = offset - (scale * mean / std_dev)
scaled_std_dev = scale / tf.sqrt(std_dev)


# In[25]:


import numpy as np
scaled_std_dev=np.array(scaled_std_dev)
print(scaled_std_dev)


# In[ ]:





# In[ ]:




