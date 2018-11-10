import numpy as np
import tensorflow as tf
from work import matlab_connection as mc

from easy_tfrecords import create_tfrecords, easy_tfrecords as records


eng = mc('desktop')

# GET SOME TEST DATA 
# - IN MATLAB, (UNLIKE NUMPY) THE FINAL AXIS IS THE INDEX AXIS
trainX = eng.get_var('ex')
trainY = eng.get_var('y')

# CREATE AND SAVE TO A COUPLE TFRECORDS FILES
create_tfrecords('data_1.tf', matlab=True, x=trainX, y=trainY)

# INSTANTIATE THE RECORDS OBJECT
batch_x, batch_y = records(['data_1.tf'], shuffle=False, batch_size=1).inputs(['x', 'y'])

with tf.Session() as sess:

  # enable batch fetchers
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for n in range(2):
    print('------------')
    print('n => {}\n'.format(n))

    x_eval, y_eval = sess.run( [batch_x, batch_y] )
    print('x_eval=\n{}\n'.format(x_eval))
    print('y_eval=\n{}'.format(y_eval))

  coord.request_stop()
  coord.join(threads)

sess.close()