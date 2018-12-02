import tensorflow as tf
import numpy as np
import json
from functools import reduce
from operator import mul
from collections import OrderedDict

DEFAULT_BATCH_SIZE = 1

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


 # TF_RECORDS_FILE => file to save, **DATA_PAIRS => key=vals of data
def create_tfrecords(tf_records_file, high_dim_sort=False, **data_pairs) :


  # WRITER FUNCTIONS ------------------------------------------

  # CREATE THE WRITER OBJECT
  writer = tf.python_io.TFRecordWriter(tf_records_file)

  # PULL THE FIRST PAIRED ARGUMENT. THIS WILL BE USED FOR SIZING PURPOSES
  indx_elem = next(iter(data_pairs), None)
  
  if high_dim_sort is True :
    num_elems = data_pairs[indx_elem].shape[-1]
    shapes_indx = [...,0]
  elif high_dim_sort is False :
    num_elems = data_pairs[indx_elem].shape[0]
    shapes_indx = [0,...]


  # ITERATE THROUGH DATA PAIRS
  for indx in range( num_elems ):

    # CREATE A {FEATURES} KEY THAT INCLUDES A FEATURE FOR EACH NAMED ARGUMENT
    example = tf.train.Example(features=tf.train.Features(feature={
      key : feature_map[val.dtype.name](val[
          [...,indx] if high_dim_sort is True else [indx,...]
        ].reshape([-1]))
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
    { 'shape' : list(val[shapes_indx].shape), 'dtype' : val.dtype.name, 'num_elems' :  reduce(mul, list(val[shapes_indx].shape), 1 ) } for key, val in data_pairs.items() 
  }

  # WRITE THE SHAPE OBJECT TO A FILE
  with open('{}.json'.format(tf_records_file), 'w') as outfile:
    json.dump(shapes, outfile)


# --------------------------------------------------- #
# FUNCTION FOR CREATING TF_RECORDS DATA
# --------------------------------------------------- #
class easy_tfrecords:

  # DATA_ID => identifier for json structure,
  def __init__(self, files=None, shuffle=False, batch_size=DEFAULT_BATCH_SIZE, keys=None) :
    
    # LOAD FILES
    self.filenames = files
    self.shuffle = shuffle
    self.batch_size = batch_size

    # GET THE BASE STRUCTURE OF THE FILE
    self.__get_base_structure()

    # INPUT KEYS
    if keys is not None :
      self.set_keys(keys)
    else :
      self.input_keys = False


  # GET BASE STRUCTURE 
  def __get_base_structure(self) :

    # LOAD JSON - DEFINE STRUCTURE FOR READER
    with open('{}.json'.format(self.filenames[0])) as data_file:
      self.base_structure = json.load(data_file)


  # READ DATA FROM THE FILE
  def __read_and_decode(self, serialized_example) :

    input_keys = self.input_keys

    features = tf.parse_single_example(
        serialized_example,

        # FEATURES WILL BE KEYS IN THE ORDER THEY WERE PASSED TO INPUTS
        features={
          key : tf.FixedLenFeature( 
            [self.base_structure[key]['num_elems']] , 
            dtype_lookup(self.base_structure[key]['dtype'])
          ) for key in input_keys

        }

        # I.E. => (e.g)
        # features={
        #     'x': tf.FixedLenFeature( [INPUT_TIME_STEPS*INPUT_DIM], tf.float32),
        #     'y': tf.FixedLenFeature( [OUTPUT_TIME_STEPS*OUTPUT_DIM], tf.float32),
        # })
    )

    # RESHAPE TO ORIGINAL DATA SHAPE
    reshape_op = dict(map(
      lambda keyvals: 
        (
          keyvals[0], 
          tf.reshape(keyvals[1], self.base_structure[keyvals[0]]['shape'])
        ), features.items()))
    
    return reshape_op


  # FUNCTION CALLED BY THE TENSORFLOW FETCH OPERATION
  def set_keys(self, input_keys) :

    if input_keys is None :
      return None
    else :
      self.input_keys = input_keys

      files = tf.data.Dataset.list_files(self.filenames)

      # INPUT PIPELINE
      # ==> https://www.tensorflow.org/guide/performance/datasets

      dataset = files.interleave(tf.data.TFRecordDataset,1)
      if self.shuffle is not False :
        dataset = dataset.shuffle(buffer_size=self.shuffle)
      dataset = dataset.map(map_func=self.__read_and_decode)
      dataset = dataset.batch(batch_size=self.batch_size)
      dataset = dataset.repeat()
      self.dataset = dataset

      # ITERATOR PIPELINE
      self.set_iterator()


  def set_iterator(self) :
    self.iterator = self.dataset.make_initializable_iterator()
    self.next_factory = self.iterator.get_next()


  def get_next_factory(self) :
    return self.next_factory


  def get_dataset(self) :
    return self.dataset


  def get_initializer(self) :
    return self.iterator.initializer

# OUTPUT THE FIRST RECORD OF THE FILE TO STRING
def tell(tf_records_file) :

  examples = tf.python_io.tf_record_iterator(tf_records_file)
  bytes = next(examples)
  example_string = '{}'.format(tf.train.Example.FromString(bytes))

  # WRITE THE SHAPE OBJECT TO A FILE
  with open('{}.txt'.format(tf_records_file), 'w') as outfile:
    outfile.write(example_string)