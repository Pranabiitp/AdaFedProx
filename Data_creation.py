#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import shutil

main_folder_path = "C:/Users/91898/Downloads/New folder"
client_paths = ["C:/Users/91898/Downloads/New folder/client1", "C:/Users/91898/Downloads/New folder/client2", "C:/Users/91898/Downloads/New folder/client3", "C:/Users/91898/Downloads/New folder/client4"]
desired_distribution = {
    'im_Dyskeratotic': [270, 156, 106, 81],
    'im_Koilocytotic': [98, 109, 121, 190],
    'im_Metaplastic': [124, 139, 89, 241],
    'im_Parabasal': [152, 84, 234, 117],
    'im_Superficial-Intermediate': [173, 131, 204, 113]
}

# Initialize client data dictionaries
client_data = {client: {} for client in client_paths}

# Iterate through the classes
for class_folder in desired_distribution:
    class_path = os.path.join(main_folder_path, class_folder)

    # Check if the item is a directory (asubfolder)
    if os.path.isdir(class_path):
        class_images = os.listdir(class_path)
        random.shuffle(class_images)  # Shuffle images randomly

        # Iterate through clients and assign images
        for i, client in enumerate(client_paths):
            start = sum(desired_distribution[class_folder][:i])
            end = sum(desired_distribution[class_folder][:i + 1])

            # Check if the client path exists in the dictionary
            if client in client_data:
                client_data[client].setdefault(class_folder, []).extend(class_images[start:end])


# Copy images to the corresponding client folders
for client, distribution in client_data.items():
    client_folder = os.path.join(client)
    os.makedirs(client_folder, exist_ok=True)

    for class_folder, images in distribution.items():
        class_path = os.path.join(main_folder_path, class_folder)
        dest_path = os.path.join(client_folder, class_folder)
        os.makedirs(dest_path, exist_ok=True)

        for image in images:
            src_path = os.path.join(class_path, image)
            dest_image_path = os.path.join(dest_path, image)
            shutil.copy(src_path, dest_image_path)

