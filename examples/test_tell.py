import numpy as np
import tensorflow as tf

from easy_tfrecords import create_tfrecords, tell


# CREATE SOME TEST DATA
x      = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.int32)
trainX = np.asarray( [x, x+1, x+2] )

y      = np.array([0.25], np.float32)
trainY = np.asarray( [y, y+1, y+2] )


# CREATE AND TFRECORDS FILE
create_tfrecords('data_1.tf', x=trainX, y=trainY)

# SAVE OUTUT STRUCTURE TO FILE
tell('data_1.tf')