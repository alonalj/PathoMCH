from shutil import copyfile
from predict import *
from network_architectures import *
import os
from tfrecords_reader import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.python.keras.utils import multi_gpu_model

'''
Control training and inference from here.
Requires tensorflow 1.14 and python 3 (specifically developed using TensorFlow 1.14.0 and python 3.6)
'''

# Train on multiple GPUs (if not available will default to 1 replica)
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# General settings
c = Conf_BRCA_TRAITS_miR_17_5p_extreme()
c.set_local()
training = True  # set to False for predictions
resample_round = 0  # can be replaced by sys.argv[..] to automate using external script
print("Resample round {}".format(resample_round))
c.APPLY_AUGMENTATIONS = True  # flip augmentations that apply only to train set
c.NETWORK_NAME = 'inception'

# Training settings
EPOCHS = 1000  # setting an upper limit. The model will likely stop before, when converging on validation set.
lr = 0.001
BATCH_SIZE_PER_REPLICA = c.BATCH_SIZE
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Prepare model folders and code snapshot used for model
model_folder = '../out/{}_zoom_{}_round_{}_{}'.format(c.NAME, c.ZOOM_LEVEL, resample_round, get_time_stamp())
os.mkdir(model_folder)
c.set_logger(model_folder)
copyfile('../src/model.py', model_folder+'/model.py')
copyfile('../src/network_architectures.py', model_folder+'/network_architectures.py')
copyfile('../src/conf.py', model_folder+'/conf.py')
model_folder_loss = model_folder + '/loss'
model_folder_acc = model_folder + '/acc'
model_folder_auc = model_folder + '/auc'
for f in [model_folder_loss, model_folder_acc, model_folder_auc]:
    if not os.path.exists(f):
        os.mkdir(f)

if training:
    # get filenames and number of tiles to determine number of steps
    training_filenames = tf.gfile.Glob(c.GCS_PATTERN.format('train/round_{}_train'.format(resample_round)))
    validation_filenames = tf.gfile.Glob(c.GCS_PATTERN.format('val/round_{}_val'.format(resample_round)))
    print("Training filenames\n", training_filenames)
    print("Val filenames:\n", validation_filenames)
    random.shuffle(training_filenames)
    random.shuffle(validation_filenames)
    # File names contain number of samples in each. Using this to obtain total number of images (tiles).
    n_train = sum([int(f.split('-')[-1].split('.')[0]) for f in training_filenames])
    n_val = sum([int(f.split('-')[-1].split('.')[0]) for f in validation_filenames])
    STEPS_PER_EPOCH_VAL = n_val // BATCH_SIZE


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


with strategy.scope():
    loss = 'binary_crossentropy'
    main_metric = tf.keras.metrics.binary_accuracy
    model = inception_keras(c)
    acc_val_metric = 'val_binary_accuracy'

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=[main_metric,lr_metric,tf.keras.metrics.AUC()])
    model.summary()

    # Train the model
    global_step = 0
    if training:
        if c.LOAD_WEIGHTS_PATH:
            model.load_weights(tf.train.latest_checkpoint(c.LOAD_WEIGHTS_PATH))
            log.print_and_log("Loaded model from path: {}".format(c.LOAD_WEIGHTS_PATH))

        TFRec = tfrecords(c)

        ckpt_prefix = "ckpt"
        checkpoint_prefix_loss = os.path.join(model_folder_loss, ckpt_prefix + "_{epoch}")
        checkpoint_prefix_acc = os.path.join(model_folder_acc, ckpt_prefix + "_{epoch}")
        checkpoint_prefix_auc = os.path.join(model_folder_auc, ckpt_prefix + "_{epoch}")
        STEPS_PER_EPOCH_TRAIN = n_train // (BATCH_SIZE * 16)  # division by batch size due to BATCH_SIZE number of tiles being processed per step. Division by 16 to evaluate every 16th epoch to avoid overfitting due to tile similarities between batches (many tiles per slide make it seem like there's a lot of the same per slide)
        if c.LOCAL:  # TODO: remove
            STEPS_PER_EPOCH_TRAIN = 2
            STEPS_PER_EPOCH_VAL = 2
        print("steps TRAIN", STEPS_PER_EPOCH_TRAIN)
        print("steps val", STEPS_PER_EPOCH_VAL)
        train_tfrecords = TFRec.get_training_dataset(training_filenames, BATCH_SIZE)
        val_tfrecords = TFRec.get_inference_dataset(validation_filenames, BATCH_SIZE)
        res = model.fit(train_tfrecords,
                        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
                        epochs=EPOCHS,
                        validation_data=val_tfrecords,
                        validation_freq=1,
                        validation_steps=STEPS_PER_EPOCH_VAL,
                        callbacks=
                        [
                            tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
                            tf.keras.callbacks.EarlyStopping(patience=30, verbose=1),
                            # default monitor for val_loss
                            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix_loss,
                                                               save_weights_only=True,
                                                               save_best_only=True),
                            tf.keras.callbacks.ModelCheckpoint(monitor='val_auc',
                                                               filepath=checkpoint_prefix_auc,
                                                               save_weights_only=True,
                                                               save_best_only=True,
                                                               mode='max')
                        ]
                    )

    else:
        if not c.LOCAL:
            assert c.LOAD_WEIGHTS_PATH, "LOAD_WEIGHTS_PATH is None!"
        conf_architecture = c
        conf_per_sample_tfrecords = c
        log.print_and_log("Inference using weights under: {}".format(c.LOAD_WEIGHTS_PATH))
        # evaluate against ground truth
        predict_per_sample(c, conf_architecture, c.LOAD_WEIGHTS_PATH, model,
                           out_folder_suffix='_round_{}'.format(resample_round))




