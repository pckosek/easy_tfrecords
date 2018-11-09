import numpy as np
import tensorflow as tf

from easy_tfrecords import create_tfrecords, easy_tfrecords as records


# CREATE SOME TEST DATA
x      = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.int32)
trainX = np.asarray( [x, x+1, x+2] )

y      = np.array([0.25], np.float32)
trainY = np.asarray( [y, y+1, y+2] )


# CREATE AND SAVE TO A COUPLE TFRECORDS FILES
create_tfrecords('data_1.tf', x=trainX, y=trainY)
create_tfrecords('data_2.tf', x=trainX+10, y=trainY+10)
create_tfrecords('data_3.tf', x=trainX+100, y=trainY+100, z=trainY+100)

# INSTANTIATE THE RECORDS OBJECT
batch_x, batch_y = records(['data_1.tf', 'data_2.tf', 'data_3.tf'], shuffle=False, batch_size=1).inputs(['x', 'y'])

with tf.Session() as sess:

  # enable batch fetchers
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for n in range(10):
    print('------------')
    print('n => {}\n'.format(n))

    x_eval, y_eval = sess.run( [batch_x, batch_y] )
    print('x_eval=\n{}\n'.format(x_eval))
    print('y_eval=\n{}'.format(y_eval))

  coord.request_stop()
  coord.join(threads)

sess.close()