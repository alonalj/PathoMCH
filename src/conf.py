from utils import Logger
import os

'''
Configurations for preprocessing and training. Conf class contains the main settings. Below Conf your will find
general BRCA and LUAD configuration classes derived from Conf. Examples of specific trait configurations (e.g. for ESR1)
can be found below. In order to complete preprocessing fast, it is recommended to set NUM_CPU=60 if available.  
Note that most paths are catered for Google Cloud buckets, but can be modified to your needs.
'''

class Conf:
    def __init__(self, is_preprocessing=False):

        self.IS_PREPROCESSING = is_preprocessing
        self.DIAGNOSTIC_SLIDES = True
        self.LOAD_WEIGHTS_PATH = None
        self.ONLY_DX1 = False
        self.ZOOM_LEVEL = 20
        self.NUM_CPU = 60
        self.SAVE_IMAGES = True
        self.RESTORE_FROM_BEST_CKPT = False
        self.APPLY_AUGMENTATIONS = True  # True if for training we want data augmentations. With coco was beneficial to first train without until plateau. Then load ckpt and retrain with.
        self.IS_TRAIN = True
        self.N_ROUNDS = 5
        self.TRAIN_PCT = 0.8
        self.VAL_PCT = 0.1
        self.NETWORK_NAME = ''
        self.USE_SAVED_LABELS_IF_EXIST = False  # false will create labels dictionary from scratch
        self.BATCH_SIZE = 18
        self.IMG_SIZE = 512
        self.VAL_STEPS_MAX = 100000
        self.NUM_CLASS = 1
        self.TASK_TYPE = '2-class'
        self.CLINICAL_LABELS = ['lo', 'hi']
        self.LABELS = self.CLINICAL_LABELS

        self.IMG_TYPE = 'jpeg'
        self.N_CHANNELS = 3

        self.N_CHAR_PATIENT_ID = 12
        self.N_CHAR_SLIDE_ID = 23
        self.N_CHAR_SAMPLE_ID = 15

        # paths
        self.OUT_DIR = '../out/'
        self.CKPT_PATH_FORMAT = "../out/model_{}"
        self.IM_PATH_TO_LABEL_DICT_FORMAT = 'im_path_to_label_dict_{}_{}'
        self.ALL_SAMPLES_TFRECORDS_FOLDER = '../res/all_samples_dummy_labels/'
        self.generate_tfrecords_folders()

        if not os.path.exists(self.OUT_DIR):
            os.mkdir(self.OUT_DIR)

        self.SVS_SLIDES_PATH = '../data/slides/diagnostic/'
        self.IMG_PATH = '../data/images/zoom_{}_{}/'.format(self.ZOOM_LEVEL, self.IMG_SIZE)
        self.SLIDE_TYPE = 'DX'

        if not os.path.exists(self.IMG_PATH) and self.IS_PREPROCESSING:
            os.makedirs(self.IMG_PATH)

        # misc
        self.PANCAN_NAME_SUFFIX = 'BRCA_UNHEALTHY_SAMPLES'

    def set_logger(self, folder_path):
        self.LOG = Logger(folder_path)

    def set_ckpt_path(self):
        self.CKPT_PATH = self.CKPT_PATH_FORMAT.format('_'.join(self.CLINICAL_LABELS))

    def generate_tfrecords_folders(self):
        for sub_data in ['train', 'val', 'all_samples_dummy_labels']:
            sub_data_path = '../res/{}/'.format(sub_data)
            if not os.path.exists(sub_data_path):
                os.mkdir(sub_data_path)

    def set_local(self):
        self.LOCAL = True
        self.GCS_PATTERN = '../res/{}*.tfrec'
        self.GCS_PATTERN_PER_SAMPLE = '../res/all_samples_dummy_labels/*tfrecords'


# presets

class Conf_BRCA(Conf):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.TCGA_COHORT_NAME = 'brca'
        self.CLINICAL_FILEPATH = '../res/clinical_data_brca.csv'
        self.GCS_PATTERN_PER_SAMPLE = 'gs://patho_al/tfrecords/brca/per_sample/tf_records_zoom_20_labels_dummy_neg_dummy_pos/*tfrecords'
        self.PANCAN_NAME_SUFFIX = 'BRCA_UNHEALTHY_SAMPLES'


class Conf_LUAD(Conf):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.TCGA_COHORT_NAME = 'luad'
        self.CLINICAL_FILEPATH = '../res/clinical_data_luad.csv'
        self.GCS_PATTERN_PER_SAMPLE = 'gs://patho_al/tfrecords/luad/per_sample/tf_records_zoom_20_labels_dummy_neg_dummy_pos/*tfrecords'
        self.PANCAN_NAME_SUFFIX = 'LUAD_UNHEALTHY_SAMPLES'


class Conf_BRCA_TRAITS_MKI67_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'MKI67_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['MKI67|4288']
        self.LOAD_WEIGHTS_PATH = None  # change when transitioning to inference, e.g.: '../out/<model_name>/auc/'
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/MKI67_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_ESR1_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'ESR1_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['ESR1|2099']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/ESR1_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_EGFR_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'EGFR_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['EGFR|1956']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/EGFR_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_FOXA1_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'FOXA1_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['FOXA1|3169']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/FOXA1_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_MYC_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'MYC_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['MYC|4609']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://main_al/tfrecords/brca/all_sharded/MYC_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_KRT14_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'KRT14_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['KRT14|3861']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://main_al/tfrecords/brca/all_sharded/KRT14_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_FOXC1_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'FOXC1_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['FOXC1|2296']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://main_al/tfrecords/brca/all_sharded/FOXC1_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_CD24_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'CD24_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['CD24|100133941']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://main_al/tfrecords/brca/all_sharded/CD24_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_ERBB2_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'ERBB2_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['ERBB2|2064']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/ERBB2_lo_vs_hi/{}*.tfrec'


class Conf_BRCA_TRAITS_miR_17_5p_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'hsa-miR-17-5p_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['hsa-miR-17-5p']
        self.LOAD_WEIGHTS_PATH = '../out/hsa-miR-17-5p_lo_vs_hi_zoom_20_round_0_2020_05_22_20_59_43/auc/'
        self.GCS_PATTERN = 'gs://main_al/tfrecords/brca/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_BRCA_TRAITS_miR_29a_3p_extreme(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'hsa-miR-29a-3p_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['hsa-miR-29a-3p']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/brca/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_TRAITS_EGFR_extreme(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'EGFR_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['EGFR|1956']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/luad/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_TRAITS_KRAS_extreme(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'KRAS_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['KRAS|3845']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/luad/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_TRAITS_miR_17_5p_extreme(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'hsa-miR-17-5p_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['hsa-miR-17-5p']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/luad/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_TRAITS_miR_21_5p_extreme(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'hsa-miR-21-5p_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['hsa-miR-21-5p']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/luad/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_TRAITS_CD274_extreme(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'CD274_lo_vs_hi'
        self.CLINICAL_LABEL_COLS = ['CD274|29126']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = 'gs://patho_al/tfrecords/luad/all_sharded/{}/'.format(self.NAME)+'{}*.tfrec'


class Conf_LUAD_DUMMY_LABEL(Conf_LUAD):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'Dummy'
        self.CLINICAL_LABEL_COLS = ['dummy']
        self.LOAD_WEIGHTS_PATH = None
        self.GCS_PATTERN = None
        self.GCS_PATTERN_PER_SAMPLE = None


class Conf_BRCA_DUMMY_LABEL(Conf_BRCA):
    def __init__(self, is_preprocessing=False):
        super().__init__(is_preprocessing)
        self.NAME = 'Dummy'
        self.CLINICAL_LABEL_COLS = ['dummy']
        self.LOAD_WEIGHTS_PATH = None
        self.PATIENT_IDS = None
        self.GCS_PATTERN_PER_SAMPLE = 'gs://patho_al/tfrecords/brca/per_sample/tf_records_zoom_20_labels_dummy_neg_dummy_pos/*tfrecords'

