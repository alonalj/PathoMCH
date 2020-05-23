from preprocess import *
from tfrecords_reader import *

'''
Functionality for predictions per sample. Used by model.py when using training = False.
'''

tf.enable_eager_execution()


class Predict(object):
    def __init__(self, c, conf_architecture, model, out_folder, weights=None):
        self.c = c
        self.conf_architeture = conf_architecture
        self.weights = weights
        self.model = model
        self.load_weights()
        self.out_folder = out_folder
        self.log = Logger('../out/', 'predict_log')

        if self.conf_architeture.TASK_TYPE == 'multilabel':
            self.n_labels = len(self.c.NAMES_CHOSEN_LABELS_ORDERED)
        elif self.conf_architeture.TASK_TYPE == '2-class':
            self.n_labels = 1

    def load_weights(self):
        if self.weights:
            if 'ckpt' in self.weights:
                self.model.load_weights(self.weights)
            else:
                self.model.load_weights(tf.train.latest_checkpoint(self.weights))
            print("Loaded weights")

    def predict_scores(self, chosen_samples):
        self.log.print_and_log("Starting prediction for {} samples.".format(len(chosen_samples)))
        labels_all_slides, names_all_slides = [], []

        for sample_tfrecord in chosen_samples:
            element_count = 0
            sample_preds, sample_scores, sample_labels, sample_names = [], [], [], []

            for element in sample_tfrecord:
                if self.n_labels == 1:
                    labels = element[1].numpy()
                else:
                    labels = element[1].numpy()[:, self.conf_architeture.IX_CHOSEN_LABELS_ORDERED]
                if len(labels) == 0:
                    break

                images = element[0].numpy()
                names = element[2].numpy()

                slide_name = names[0].decode("utf-8")[:self.c.N_CHAR_SLIDE_ID]

                if os.path.exists("{}/scores_{}.pkl".format(self.out_folder, slide_name)):
                    self.log.print_and_log("Found scores for {}. Continuing.".format(slide_name))
                    break

                batch_x, batch_y = images, labels
                sample_scores_batch = self.model.predict(batch_x)

                if len(sample_scores) == 0:
                    sample_scores = sample_scores_batch
                    sample_labels = batch_y
                    sample_names = names
                else:
                    sample_scores = np.append(sample_scores, sample_scores_batch, 0)
                    sample_labels = np.append(sample_labels, batch_y, 0)
                    sample_names = np.append(sample_names, names, 0)

                if self.c.LOCAL and element_count == 5:
                    break
                element_count += 1

            if len(labels) == 0:
                continue

            if os.path.exists("{}/scores_{}.pkl".format(self.out_folder, slide_name)):
                continue

            names_all_slides.append(slide_name)

            save_obj(sample_scores, "scores_{}".format(slide_name), self.out_folder)
            save_obj(sample_labels, "labels_{}".format(slide_name), self.out_folder)
            save_obj(sample_names, "names_{}".format(slide_name), self.out_folder)

        save_obj(names_all_slides, 'names_available_slides', self.out_folder)


def get_per_sample_tfrecords(c, sample_tfrecord_paths):
    TFRecords = tfrecords(c, with_name=True)

    per_sample_tfrecords = []

    for s in sample_tfrecord_paths:
        sample_data = TFRecords.get_inference_dataset([s], c.BATCH_SIZE, repeat=False)
        per_sample_tfrecords.append(sample_data)

    return per_sample_tfrecords


def predict_per_sample(c, conf_architecture, weights_path, model_architecture, out_folder_suffix=''):
    tfrecords_per_sample = get_per_sample_tfrecords(c, tf.gfile.Glob(c.GCS_PATTERN_PER_SAMPLE))

    out_folder = '../out/predict/{}/'.format(c.NAME + out_folder_suffix)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    evaluator = Predict(c, conf_architecture, model_architecture, out_folder)

    evaluator.weights = weights_path
    evaluator.load_weights()
    evaluator.predict_scores(tfrecords_per_sample)