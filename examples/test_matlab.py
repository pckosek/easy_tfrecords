import numpy as np
import tensorflow as tf
from work import matlab_connection as mc
from easy_tfrecords import create_tfrecords, easy_tfrecords as records


eng = mc('desktop')

# GET SOME TEST DATA 
trainX = eng.get_var('ex')
trainY = eng.get_var('y')

# CREATE AND SAVE TO A COUPLE TFRECORDS FILES
# - IN MATLAB, (UNLIKE NUMPY) THE FINAL AXIS IS THE INDEX AXIS 
#   => high_dim_sort=True
create_tfrecords('data_1.tf', high_dim_sort=True, x=trainX, y=trainY)

# INSTANTIATE THE RECORDS OBJECT
rec = records(files=['data_1.tf', 'data_2.tf'],
  shuffle=False,
  batch_size=1, 
  keys=['x', 'y'])

next_factory = rec.get_next_factory()
batch_x = next_factory['x']
batch_y = next_factory['y']

with tf.Session() as sess:

  sess.run(rec.get_initializer())

  for n in range(2):
    print('------------')
    print('n => {}\n'.format(n))

    x_eval, y_eval = sess.run( [batch_x, batch_y] )
    print('x_eval=\n{}\n'.format(x_eval))
    print('y_eval=\n{}'.format(y_eval))

sess.close()