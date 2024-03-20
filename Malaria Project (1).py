#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from PIL import Image


# In[3]:


# Specify the directory containing the images
directory_Parasitized = "D:/Black Coffer/archive/archive1/cell_images/Parasitized/"
directory_uninfected = "D:/Black Coffer/archive/archive1/cell_images/Uninfected/"

# List all files in the directory
files_Parasitized = os.listdir(directory_Parasitized)
files_uninfected = os.listdir(directory_uninfected)
files_total = files_uninfected+files_Parasitized


# In[4]:


# Initialize a counter for the number of images
num_images_Parasitized = 0
num_images_uninfected = 0
num_images_total = 0

# Iterate through each file in the directory
for file in files_Parasitized:
    # Check if the file is an image (you can add more image file extensions if needed)
    if file.endswith(".png"):
        num_images_Parasitized += 1

for file in files_uninfected:
    if file.endswith(".png"):
        num_images_uninfected += 1

for file in files_total:
    if file.endswith(".png"):
        num_images_total += 1
    
# Print the number of images
print("Number of Parasitized images:", num_images_Parasitized)
print("Number of Uninfected images:", num_images_uninfected)
print("Number of Total images:", num_images_total)


# In[5]:


print(len(files_Parasitized))
print(files_Parasitized)


# In[6]:


print(len(files_uninfected))
print(files_uninfected)


# In[7]:


import numpy as np
from PIL import Image
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


# In[8]:


#Displaying parasitized cell image
img = mpimg.imread('D:/Black Coffer/archive/archive1/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png')
plt.imshow(img)
plt.show()


# In[9]:


#Displaying uninfected cell image
img = mpimg.imread('D:/Black Coffer/archive/archive1/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png')
plt.imshow(img)
plt.show()


# In[10]:


#Resizing images and copying to a new file

#New file directory
resized_images_file = "D:/Black Coffer/archive/archive1/cell_images/zResized_Images/"


for i in range(1000):
    filename = files_Parasitized[i]
    input_img_path = os.path.join(directory_Parasitized, filename)
    img = Image.open(input_img_path)
    img = img.resize((96, 96)).convert('RGB')
    
    new_filename = f"Parasitized_{i}.jpg"
    output_img_path = os.path.join(resized_images_file, new_filename)
    img.save(output_img_path)
    img.close()

for i in range(1000):
    filename = files_uninfected[i]
    input_img_path = os.path.join(directory_uninfected, filename)
    img = Image.open(input_img_path)
    img = img.resize((96, 96)).convert('RGB')
    
    new_filename = f"Uninfected_{i}.jpg"
    output_img_path = os.path.join(resized_images_file, new_filename)
    img.save(output_img_path)
    img.close()


# In[11]:


fresized_images_file = os.listdir(resized_images_file)
fresized_images_file


# In[12]:


print(len(fresized_images_file))


# In[13]:


#Displaying Resized parasitized cell image
img1 = mpimg.imread('D:/Black Coffer/archive/archive1/cell_images/zResized_Images/Parasitized_0.jpg')
plt.imshow(img1)
plt.show()


# In[14]:


#Displaying Resized Uninfected cell image
img1 = mpimg.imread('D:/Black Coffer/archive/archive1/cell_images/zResized_Images/Uninfected_0.jpg')
plt.imshow(img1)
plt.show()


# In[15]:


#Creating labels
#Parasitized cells is labelled as 1 & Uninfected Cells is labelled as 0
labels = []
Parasitized_cell = 0
Uninfected_cell = 0 
for i in range(2000):
    file_name = fresized_images_file[i]
    label = file_name[0:10]
    if label == "Parasitize":
        labels.append(1)
        Parasitized_cell += 1
    else:
        labels.append(0)
        Uninfected_cell += 1


# In[16]:


print(labels[0:5])
print(len(labels))
print(Parasitized_cell)#No. of Parasitized cells
print(Uninfected_cell)#No. of Uninfected Cells


# In[17]:


import cv2
import glob


# In[18]:


#Converting to Numpy Arrays
#resized_images_file
image_extension = ['png', 'jpg']
files = []
[files.extend(glob.glob(resized_images_file + '*.' + e)) for e in image_extension]
para_uninfec_images = np.asarray([cv2.imread(file) for file in files])


# In[19]:


print(type(para_uninfec_images))


# In[20]:


X = para_uninfec_images
Y = np.asarray(labels)


# In[21]:


#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[22]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[23]:


#Scaling
X_trained_scaled = X_train/255
X_test_scaled = X_test/255


# In[24]:


X_trained_scaled


# In[25]:


#Building Nueral Networks
import tensorflow as tf
import tensorflow_hub as hub


# In[26]:


mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape = (96, 96,3), trainable = False)


# In[27]:


# Define the URL of the MobileNet V2 model from TensorFlow Hub
module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/5"

# Create a Keras Lambda layer using the TensorFlow Hub KerasLayer
hub_layer = hub.KerasLayer(module_url, input_shape=(96, 96, 3))

# Wrap the TensorFlow Hub KerasLayer inside a Keras Lambda layer
lambda_layer = tf.keras.layers.Lambda(lambda x: hub_layer(x))

# Create a Sequential model
model = tf.keras.Sequential([
    lambda_layer,
    tf.keras.layers.Dense(2)  # Output layer with 2 units for binary classification
])

# Display the model summary
model.summary()


# In[28]:


# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model to some data (replace X_train and y_train with your actual data)
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))

# Print the model summary
print(model.summary())


# In[29]:


model.save("D:/Black Coffer/archive/model.h5")


# In[30]:


score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)


# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

input_image_path = input('Path of the image to be predicted: ')
input_image = cv2.imread(input_image_path)

# Display the input image
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis
plt.show()

input_image_resize = cv2.resize(input_image, (96, 96))
input_image_scaled = input_image_resize / 255
image_reshaped = np.reshape(input_image_scaled, [1, 96, 96, 3])

input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)
print(input_pred_label)

if input_pred_label == 1:
    print('The image represents a Parasitized Cell')
else:
    print('The image represents an Uninfected Cell')


# In[ ]:






