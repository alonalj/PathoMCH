import tensorflow as tf
# import random
import random

AUTO = tf.data.experimental.AUTOTUNE
tf.compat.v1.enable_eager_execution()


class tfrecords():
    def __init__(self, c, with_name=False):
        self.c = c
        self.with_name = with_name

    def augment(self, image):
        image = tf.image.random_flip_left_right(image, seed=35155)
        image = tf.image.random_flip_up_down(image, seed=35155)
        # image = tf.image.random_brightness(image, max_delta=32. / 255.)
        # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        # image = tf.image.random_hue(image, max_delta=0.05)
        # image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        # image = tf.image.random_crop(image, [self.c.IMG_SIZE - 20, self.c.IMG_SIZE - 20, 3])
        # image = tf.image.resize(image, [self.c.IMG_SIZE, self.c.IMG_SIZE],
        #                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def read_tfrecord(self, example, is_train=True):
        if self.c.TASK_TYPE:
            n_class = self.c.NUM_CLASS
        else:
            n_class = 1
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([n_class], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_name':tf.FixedLenFeature([], tf.string),
        }
        # decode the TFRecord
        example = tf.parse_single_example(example, features)
        image = tf.image.decode_jpeg(example['image_raw'], channels=3)
        image = tf.image.resize(image, (self.c.IMG_SIZE, self.c.IMG_SIZE))
        image = tf.cast(image, tf.float32)

        # grayscale
        if self.c.TASK_TYPE == 'autoencode':
            image_gray = tf.image.rgb_to_grayscale(image)
            image_gray = image_gray / 255.
            image_gray = tf.image.resize(image_gray, (self.c.IMG_SIZE // 4, self.c.IMG_SIZE // 4))

        # normalize
        image = image / 255. #(image - self.c.MEAN) / self.c.STD
        if self.c.APPLY_AUGMENTATIONS and is_train:
            image = self.augment(image)

        image = tf.image.resize(image, [self.c.IMG_SIZE, self.c.IMG_SIZE],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self.c.TASK_TYPE == 'autoencode':
            label = image_gray
        else:
            label = example['label']
        label = tf.cast(label, tf.float32)
        if self.with_name:
            name = example['image_name']
            return image, label, name
        return image, label  # During training -- just image and label


    def read_tfrecord_inference(self, example):
        return self.read_tfrecord(example, is_train=False)


    def _load_dataset(self, filenames, is_train=True):
      # read from TFRecords. For optimal performance, use "interleave(tf.data.TFRecordDataset, ...)"
      # to read from multiple TFRecord files at once and set the option experimental_deterministic = False
      # to allow order-altering optimizations.

      option_no_order = tf.data.Options()
      option_no_order.experimental_deterministic = False

      dataset = tf.data.Dataset.from_tensor_slices(filenames)
      dataset = dataset.with_options(option_no_order)
      if is_train:
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO) # faster
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTO)
      else:
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO) # faster
        dataset = dataset.map(self.read_tfrecord_inference, num_parallel_calls=AUTO)
      return dataset


    def get_batched_dataset(self, filenames, batch_size):
      random.shuffle(filenames)
      dataset = self._load_dataset(filenames)
      # dataset = dataset.cache() # This dataset fits in RAM
      # dataset = dataset.shuffle(2435) # TODO
      dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder will be needed on TPU
      dataset = dataset.repeat()
      dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
      # For proper ordering of map/batch/repeat/prefetch, see Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets
      return dataset

    def get_batched_dataset_inference(self, filenames, batch_size, repeat=True):
      dataset = self._load_dataset(filenames, is_train=False)
      dataset = dataset.batch(batch_size)
      if repeat:
        dataset = dataset.repeat()
      # dataset = dataset.cache() # This dataset fits in RAM
      dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
      # For proper ordering of map/batch/repeat/prefetch, see Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets
      return dataset

    def get_training_dataset(self, training_filenames, batch_size):
      return self.get_batched_dataset(training_filenames, batch_size)

    def get_inference_dataset(self, inference_filenames, batch_size, repeat=True):
      return self.get_batched_dataset_inference(inference_filenames, batch_size, repeat)
