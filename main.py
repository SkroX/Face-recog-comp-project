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

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


def verify(image_path, identity, database, model):
    
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    
    if dist < 0.7:
        print("It's " + str(identity))
        door_open = True
    else:
        print("It's not " + str(identity))
        door_open = False

    return door_open


verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)

yourpath = '/content/Face-Recognition/images'

import os
import shutil
for root, dirs, files in os.walk(yourpath, topdown=False):
  for name in files:
    print(name)
    if (verify('/content/Face-Recognition/images/' + str(name), "younes", database, FRmodel)):
      shutil.move('/content/Face-Recognition/images/' + str(name), "/content/Face-Recognition/images2/")



# def who_is_it(image_path, database, model):
#     encoding = img_to_encoding(image_path, model)    
#     min_dist = 10000
#     for (name, x) in database.items():
#         dist = np.linalg.norm(encoding - x)
#         if dist < min_dist:
#             min_dist = dist
#             identity = name

#     if min_dist > 0.7:
#         print("Not in the database.")
#     else:
#         print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
#     return min_dist, identity

# who_is_it("images/camera_0.jpg", database, FRmodel)







