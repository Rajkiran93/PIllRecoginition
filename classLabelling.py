#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import pickle
import random
import numpy as np
#import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X_train = []
y_labels = []
BASE_DIR = "Agumented Data"
IMG_SIZE = 256
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg") or file.lower().endswith(".png") or file.lower().endswith("r=300"):
            path = os.path.join(root,file)
            image = cv2.imread(path)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            if os.path.basename(root) == "Abilify 10":
                label = 0
            elif os.path.basename(root) == "Abilify 15":
                label = 1
            elif os.path.basename(root) == "Abilify 20":
                label = 2
            elif os.path.basename(root) == "Abilify 30":
                label = 3
            elif os.path.basename(root) == "Almox D":
                label = 4
            elif os.path.basename(root) == "Aspirin":
                label = 5
            elif os.path.basename(root) == "Dolo":
                label = 6
            elif os.path.basename(root) == "Eldoper":
                label = 7
            elif os.path.basename(root) == "Glucomust":
                label = 8
            elif os.path.basename(root) == "GlutaonD":
                label = 9
            elif os.path.basename(root) == "Paramet Black":
                label = 10
            elif os.path.basename(root) == "Wellbutrin 100":
                label = 11
            elif os.path.basename(root) == "Wellbutrin 150":
                label = 12
            else:
                label = 13
            X_train.append([gray, label])
            
random.shuffle(X_train)
X = []
y = []
for feature, label in X_train:
    X.append(feature)
    y.append(label)

print("X->",len(X))
data = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(np.shape(data))

print(len(data), len(y))

pickle_out = open("Agumented Data/X.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()
pickle_out = open("Agumented Data/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[3]:


import tarfile
tar = tarfile.open("pickle.tar", "w:tar")
for name in ["X.pickle", "y.pickle"]:
    tar.add("Agumented Data/"+name)
tar.close()

