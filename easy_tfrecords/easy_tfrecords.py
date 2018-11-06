import tensorflow as tf
import numpy as np
import json
from functools import reduce
from operator import mul

BATCH_SIZE = 1

# --------------------------------------------------- #
# HELPER FUNCTIONS FOR TYPES AND LISTS OF TYPES
# (https://github.com/bgshih/seglink/blob/master/tool/create_datasets.py)
# these will be used in creating tfrecords
# --------------------------------------------------- #
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

feature_map = {
  'int32'   : _int64_list_feature,
  'uint32'  : _int64_list_feature,
  'int64'   : _int64_list_feature,
  'unt64'   : _int64_list_feature,
  'float32' : _float_list_feature,
  'float64' : _float_list_feature
}  


# THIS PROVUDES CUSTOM MAPPING FOR DTYPES.
 # - SPECIFICALLY, float64 to float32s for the sake of efficiency
def dtype_lookup(dtype_in) :
  return {
      'int32': tf.int64,
      'int64': tf.int64,
      'float32': tf.float32,
      'float64': tf.float32
  }[dtype_in]

# --------------------------------------------------- #
# FUNCTION FOR CREATING TF_RECORDS DATA
# --------------------------------------------------- #
class easy_tfrecords:

  # DATA_ID => identifier for json structure,
  def __init__(self, data_id) :
    self.data_id    = data_id
    

   # TF_RECORDS_FILE => file to save, **DATA_PAIRS => key=vals of data
  def create_tfrecords(self, tf_records_file, **data_pairs) :

    # WRITER FUNCTIONS ------------------------------------------

    # CREATE THE WRITER OBJECT
    writer = tf.python_io.TFRecordWriter(tf_records_file)

    # PULL THE FIRST PAIRED ARGUMENT. THIS WILL BE USED FOR SIZING PURPOSES
    indx_elem = next(iter(data_pairs), None)
    num_elems = data_pairs[indx_elem].shape[0]

    # ITERATE THROUGH DATA PAIRS
    for indx in range( num_elems ):

      # CREATE A {FEATURES} KEY THAT INCLUDES A FEATURE FOR EACH NAMED ARGUMENT
      example = tf.train.Example(features=tf.train.Features(feature={
        key : feature_map[val.dtype.name](val[indx].reshape([-1]))
          for key, val in data_pairs.items()
      }))

      # I.E. => (e.g)  example = tf.train.Example(features=tf.train.Features(feature={
      #                  'x': _float_list_feature(thisX),
      #                  'y': _float_list_feature(thisY),
      #                }))

      # WRITE THIS EXAMPLE TO THE FILE
      writer.write(example.SerializeToString())  

    # CLOSE THE WRITER OBJECT
    writer.close()

    # DATA SHAPE REFERENCE FUNTIONS ---------------------------

    # DETERMINE THE SHAPE OF EACH KEYED VALUE
    shapes = { key : 
      { 'shape' : list(val[0].shape), 'dtype' : val.dtype.name, 'num_elems' :  reduce(mul, list(val[0].shape), 1 ) } for key, val in data_pairs.items() 
    }

    # WRITE THE SHAPE OBJECT TO A FILE
    with open('{}.json'.format(self.data_id), 'w') as outfile:
      json.dump(shapes, outfile)


  # SET BASE STRUCTURE 
  def get_base_structure(self) :
    # LOAD JSON - THIS WILL NEED TO GO IN THE CLASS
    with open('{}.json'.format(self.data_id)) as data_file:
      self.base_structure = json.load(data_file)


  # READ DATA FROM THE FILE
  def read_and_decode(self, filename_queue, shuffle=False, batch_size=BATCH_SIZE) :

    # CONSTRUCT OBJECT READER
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={ 
          key : tf.FixedLenFeature([val['num_elems']], dtype_lookup(val['dtype']) )
            for key, val in self.base_structure.items()
        }

        # I.E. => (e.g)
        # features={
        #     'x': tf.FixedLenFeature( [INPUT_TIME_STEPS*INPUT_DIM], tf.float32),
        #     'y': tf.FixedLenFeature( [OUTPUT_TIME_STEPS*OUTPUT_DIM], tf.float32),
        # })
    )

    # RESHAPE TO ORIGINAL DATA SHAPE
    reshape_op = [ tf.reshape(val, self.base_structure[key]['shape']) for key, val in features.items() ]

    # BATCH OPTIONALLY SHUFFLED DATA
    if shuffle==True :
      possibly_batched = tf.train.shuffle_batch(
        reshape_op, 
        batch_size=batch_size,
        capacity=100,
        min_after_dequeue=20)
    elif shuffle==False :
      possibly_batched = tf.train.batch(
        reshape_op, 
        batch_size=batch_size)

    return possibly_batched


  # FUNCTION CALLED BY THE TENSORFLOW FETCH OPERATION
  def inputs(self, filenames, shuffle=False, batch_size=BATCH_SIZE) :

    # LOAD APPROPRIATE DATA STRUCTURE - IF NECESSARY
    try:
        self.base_structure
    except :
        self.get_base_structure()    
    
    # LOAD FILES
    filename_queue = tf.train.string_input_producer(filenames)
    return self.read_and_decode(filename_queue, shuffle, batch_size)

    # GIVE THE INPUTS NAMES
    # x = tf.identity(x, name="x_batch_{}".format(name))
    # y = tf.identity(y, name="y_batch_{}".format(name))

    # return x, y
