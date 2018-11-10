# easy_tfrecords

### this package is designed to assist reading and writing to tfrecord files in an intuitive way that preserves dtype and data structure

### _Purpose_:<br>
The tfrecord format is a fast and powerful way of feeding data to a tensorflow model; it can automatically batch, randomize and iterate your data across epochs without special instructions. The **problem** with using tfrecord files comes from orchestrating the madness of matching feature structures across the reader, writer and fetcher.
<br><br>
The **easy_tfrecords** module contains methods and classes that allow you to write to and read from tfrecord files in a straightforward, extensible manner.

### _Features_:<br>
- create tfrecord files
- read from single or multiple tfrecord files
- selectively read data from tfrecord files
- examine the data structure of tfrecord files

### _Usage_:<br>
#### **Writing**<br>
- Import data into python however you normally would (excel, pandas, csv, matlab, etc.)
- Reshape each of your arrays of features to `shape=[N, x[, y[, z[, etc.]]]]` where N is the number of features. 
  - Add multiple lists of features to the file as key-value pairs
#### **Reading**<br>
- Create a reader class object, specifying your file list (can be length 1), optionally specifying batch size and shuffe spec.
- pass a list of which inputs to read from the file

#### Example Code:
```python
import numpy as np
import tensorflow as tf

from easy_tfrecords import create_tfrecords, easy_tfrecords as records


# CREATE SOME TEST DATA
x      = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.int32)
trainX = np.asarray( [x, x+1, x+2] )

y      = np.array([0.25], np.float32)
trainY = np.asarray( [y, y+1, y+2] )


# CREATE AND SAVE TO A FEW TFRECORDS FILES
create_tfrecords('tfr_1.tf', x=trainX, y=trainY)
create_tfrecords('tfr_2.tf', x=trainX+10, y=trainY+10)
create_tfrecords('tfr_3.tf', x=trainX+100, y=trainY+100, z=trainY+100)

# INSTANTIATE THE RECORDS OBJECT
batch_x, batch_y = records(['tfr_1.tf', 'tfr_2.tf', 'tfr_3.tf'], shuffle=False, batch_size=1).inputs(['x', 'y'])

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
```
#### Output :
```
------------
n => 0

x_eval=
[[ 0.25]]

y_eval=
[[[0 0 0 0]
  [0 0 0 0]]]
------------
n => 1

x_eval=
[[ 1.25]]

y_eval=
[[[1 1 1 1]
  [1 1 1 1]]]
------------
n => 2

x_eval=
[[ 2.25]]

y_eval=
[[[2 2 2 2]
  [2 2 2 2]]]
------------
n => 3

x_eval=
[[ 100.25]]

y_eval=
[[[100 100 100 100]
  [100 100 100 100]]]
------------
n => 4

x_eval=
[[ 101.25]]

y_eval=
[[[101 101 101 101]
  [101 101 101 101]]]
------------
n => 5

x_eval=
[[ 102.25]]

y_eval=
[[[102 102 102 102]
  [102 102 102 102]]]
------------
n => 6

x_eval=
[[ 10.25]]

y_eval=
[[[10 10 10 10]
  [10 10 10 10]]]
------------
n => 7

x_eval=
[[ 11.25]]

y_eval=
[[[11 11 11 11]
  [11 11 11 11]]]
------------
n => 8

x_eval=
[[ 12.25]]

y_eval=
[[[12 12 12 12]
  [12 12 12 12]]]
------------
n => 9

x_eval=
[[ 0.25]]

y_eval=
[[[0 0 0 0]
  [0 0 0 0]]]
```