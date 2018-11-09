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
- Import data into python however you normally would (excel, pandas, csv, matlab, etc.)
- Reshape each of your arrays of features to `shape=[N, x[, y[, z[, etc.]]]]` where N is the number of features

#### Example Code:

```python
import numpy as np

# CREATE SOME TEST DATA
x      = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], np.int32)
trainX = np.asarray( [x, x+1, x+2] )

y      = np.array([[10.5]], np.float32)
trainY = np.asarray( [y-1,y,y+1] )

print('trainX = \n{}\n'.format(trainX))
print('trainY = \n{}'.format(trainY))
```

    trainX = 
    [[[0 0 0 0]
      [0 0 0 0]]
    
     [[1 1 1 1]
      [1 1 1 1]]
    
     [[2 2 2 2]
      [2 2 2 2]]]
    
    trainY = 
    [[[  9.5]]
    
     [[ 10.5]]
    
     [[ 11.5]]]
    
