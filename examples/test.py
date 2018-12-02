import numpy as np
import tensorflow as tf
from easy_tfrecords import create_tfrecords, easy_tfrecords as records


# CREATE SOME TEST DATA
x      = np.array([0], np.int32)
trainX = np.asarray( [x, x+1, x+2] )

y      = np.array([10.5], np.float32)
trainY = np.asarray( [y, y+1, y+2] )

f      = np.array([100.3, 100.4], np.float32)
trainF = np.asarray( [f, f+1, f+2] )


# CREATE AND SAVE TO A FEW TFRECORDS FILES
create_tfrecords('data_1.tf', x=trainX, y=trainY, f=trainF)
create_tfrecords('data_2.tf', x=trainX+5, y=trainY+5, f=trainF+5)

# INSTANTIATE THE RECORDS OBJECT
rec = records(files=['data_1.tf', 'data_2.tf'],
  shuffle=False,
  batch_size=1, 
  keys=['x', 'y'])

next_factory = rec.get_next_factory()

batch_x = next_factory['x']
batch_y = next_factory['y']
# batch_f = next_factory['f']


with tf.Session() as sess:

  sess.run(rec.get_initializer())

  for n in range(5):
    print('-----------------------------------')
    print('n => {}\n'.format(n))

    x_eval, y_eval = sess.run( [batch_x, batch_y] )
    print('x_eval => {}'.format(x_eval))
    print('y_eval => {}'.format(y_eval))

    # x_eval, y_eval, f_eval = sess.run( [batch_x, batch_y, batch_f] )
    # print('x_eval => {}'.format(x_eval))
    # print('y_eval => {}'.format(y_eval))
    # print('f_eval => {}'.format(f_eval))


sess.close()


