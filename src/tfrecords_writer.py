import math
from utils import *
import multiprocessing
print("Tensorflow version " + tf.__version__)
tf.enable_eager_execution()
import glob
import functools
import random

log = Logger('../out/')
'''
Based on: 
https://colab.research.google.com/github/GoogleCloudPlatform/training-data-analyst/blob/master/courses/fast-and-lean-data-science/03_Flower_pictures_to_TFRecords.ipynb
Heavily adapted.
'''

def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class tfrecords_writer():
    def __init__(self, c):
        self.c = c
        self.GCS_PATTERN = '../data/{}/*' + c.IMG_TYPE
        self.GCS_OUTPUT = '../res/'

    def _decode_image(self, filename):
      bits = tf.read_file(filename)
      if self.c.IMG_TYPE == 'jpeg':
        image = tf.image.decode_jpeg(bits)
      else:
        image = tf.image.decode_png(bits)

      return image

    def resize_and_crop_image(self, image):
        image = tf.image.resize_image_with_pad(image, self.c.IMG_SIZE,self.c.IMG_SIZE)
        return image

    def recompress_image(self, image):
      image = tf.cast(image, tf.uint8)
      image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
      return image
    
    def image_example(self, image_string, label, image_name):
        if self.c.IMG_TYPE == 'jpeg':
            image_shape = tf.image.decode_jpeg(image_string).shape
        elif self.c.IMG_TYPE == 'png':
            image_shape = tf.image.decode_png(image_string).shape

        label_feature = _int64_feature(label)

        feature = {
            'height': _int64_feature([image_shape[0]]),
            'width': _int64_feature([image_shape[1]]),
            'depth': _int64_feature([image_shape[2]]),
            'label': label_feature,
            'image_raw': _bytes_feature(image_string),
            'image_name': _bytes_feature(image_name.encode('utf-8')),
        }
    
        return tf.train.Example(features=tf.train.Features(feature=feature))
    
    def save_tfrecords(self, sub_data_items, tfrecords_name):
        with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
            for filepath, label in sub_data_items:
                image = self._decode_image(filepath)
                image = self.resize_and_crop_image(image)
                image_string = self.recompress_image(image)
                image_string = image_string.numpy()
                filename = filepath.split('/')[-1]
                tf_example = self.image_example(image_string, label, filename)
                writer.write(tf_example.SerializeToString())
                # for line in str(image_example(image_string, label, filename)).split('\n')[:15]:
                #     print(line)
                # print('...')
    
    def create_single_tfrecord_shard(self, shard_id, shard_id_to_shard_filenames, im_path_to_label_dict,
                                     sub_data_name, resample_round):
        shard_items = []
        shard_size = 0
        for img_path in shard_id_to_shard_filenames[shard_id]:
            try:
                image_key =self.c.IMG_PATH + img_path.split('/')[-1]
                label = im_path_to_label_dict[image_key]
            except:
                continue
            shard_items.append((img_path, label))
            shard_size += 1

        if sub_data_name == 'test':
            tfrecords_name = self.GCS_OUTPUT +"{}/{}_{:02d}-{}.tfrec"\
                .format(sub_data_name, sub_data_name, shard_id, shard_size)
        else:
            tfrecords_name = self.GCS_OUTPUT + "{}/round_{}_{}_{:02d}-{}.tfrec"\
                .format(sub_data_name, resample_round, sub_data_name, shard_id, shard_size)
        self.save_tfrecords(shard_items, tfrecords_name)
    
    def create_sharded_tfrecords_multiprocess(self, data_name, data_filepaths, resample_round):

        im_path_to_label_dict = load_obj(self.c.IM_PATH_TO_LABEL_DICT_FORMAT.format(self.c.SLIDE_TYPE, '_'.join(self.c.LABELS)))
    
        test_img_paths = glob.glob(self.TEST_IMG_PATH)
        train_img_paths = glob.glob(self.TRAIN_IMG_PATH)
        val_img_paths = glob.glob(self.VAL_IMG_PATH)
    
        random.shuffle(test_img_paths)
        random.shuffle(train_img_paths)
        random.shuffle(val_img_paths)

        log.print_and_log(data_name)
        if "train" in data_name:
            SHARDS = 100
        else:
            SHARDS = 20

        nb_images = len(data_filepaths)
        log.print_and_log("{} has {} images".format(data_name, nb_images))
        shard_size = math.ceil(1.0 * nb_images / SHARDS)

        shard_id_to_shard_filenames = {}
        for i in range(SHARDS):
            shard_id_to_shard_filenames[i] = data_filepaths[i*shard_size:(i+1)*shard_size]

        print("round", resample_round)
        pool = multiprocessing.Pool(self.c.NUM_CPU)
        pool.map(functools.partial(self.create_single_tfrecord_shard,
                                   shard_id_to_shard_filenames=shard_id_to_shard_filenames,
                                   im_path_to_label_dict=im_path_to_label_dict,
                                   sub_data_name=data_name,
                                   resample_round=resample_round
                                   ),
                       shard_id_to_shard_filenames.keys())
        pool.close()
        pool.terminate()
        pool.join()

