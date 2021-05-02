import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from fr_utils import *
from inception_blocks_v2 import *
import os
import shutil
import h5py
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import cv2
import warnings
warnings.filterwarnings("ignore")

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))
    
    return loss

with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.compat.v1.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.compat.v1.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.compat.v1.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = triplet_loss(y_true, y_pred)
    
    print("loss = " + str(loss.eval()))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

def classify(image_path, database, model):
    
    encoding = img_to_encoding(image_path, model)
    maxi = 100000
    classe = ""
    for (u, v) in database.items():
      dist = np.linalg.norm(encoding - database[u])
      if dist < maxi:
        classe = u
        maxi = dist
    return classe

def accuracy(x, y, database, model):
  count = 0
  
  for i in range(len(x)):
    cv2.imwrite("temp.jpg",x[i])
    if classify("temp.jpg", database, model) == y[i]:
      count += 1
  print("\nAccuracy is: " + str((count * 100)/ len(x)) + " %\n")


hf = h5py.File('datasets/train_face.h5', 'r')

label = hf.get('list_classes')
label = np.array(label)

x = hf.get('train_set_x')
x = np.array(x)

y = hf.get('train_set_y')
y = np.array(y)
database = {}
for i in range(len(x)):
  cv2.imwrite("temp.jpg",x[i])
  database[y[i]] = img_to_encoding("temp.jpg", FRmodel)

y_pred = []
for i in range(len(x)):
  cv2.imwrite("temp.jpg",x[i])
  y_pred.append(classify("temp.jpg", database, FRmodel))

accuracy(x, y, database, FRmodel)
cm = confusion_matrix(y, y_pred, label)

print("Printing the Confusion Matrix\n")
print(cm)
print("Plotting the confusion matrix\n")
df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGH"],
                  columns = [i for i in "ABCDEFGH"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()
database = {}

database["shubham"] = img_to_encoding("images/sk1.jpg", FRmodel)
database["rdj"] = img_to_encoding("images/rdj.png", FRmodel)
database["chris"] = img_to_encoding("images/chris.png", FRmodel)
database["scarlett"] = img_to_encoding("images/scarlett.png", FRmodel)


for key in database:
    path = "/content/Face-recog-comp-project/" + key
    os.mkdir(path) 

def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    
    if dist < 0.7:
        door_open = True
    else:
        door_open = False

    return door_open


yourpath = '/content/Face-recog-comp-project/images'


for root, dirs, files in os.walk(yourpath, topdown=False):
  for name in files:
    print(name)
    for key in database:
        if (verify('/content/Face-recog-comp-project/images/' + str(name), key, database, FRmodel)):
            shutil.move('/content/Face-recog-comp-project/images/' + str(name), "/content/Face-recog-comp-project/" + key + "/")
            break






