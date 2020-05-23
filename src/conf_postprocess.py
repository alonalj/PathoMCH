from utils import load_obj


class Conf_Postprocess():
    def __init__(self, c):
        self.c = c
        import os
        self.OUT_DIR = '../out/{}/'.format(self.c.TCGA_COHORT_NAME)
        self.RES_DIR = '../res/postprocess/{}/'.format(self.c.TCGA_COHORT_NAME)
        for p in [self.OUT_DIR, self.RES_DIR]:
            if not os.path.exists(p):
                os.makedirs(p)

        self._model_name = self.c.NAME
        self.MODEL_DIR = '{}/'.format(self._model_name)
        self.PREDS_DIR = self.RES_DIR + 'preds/'
        self._data_split_dir = self.RES_DIR + 'data_splits/'
        self.OUT_DIR_CARTO = self.OUT_DIR + 'cartography/regular/'
        self._make_paths()

    def _make_paths(self):
        self._model_name_with_resample_round = self._model_name + '_round_{}/'
        self.MODEL_PREDS_DIR = self.PREDS_DIR + self.MODEL_DIR
        self.DATA_SPLIT_DIR = self._data_split_dir + self.MODEL_DIR
        self.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND = self.PREDS_DIR + self.MODEL_DIR + self._model_name_with_resample_round

    def get_test_patients(self):
        patients_test = load_obj('test_img_paths_DX', self.DATA_SPLIT_DIR + 'test/')
        patients_test = list(set([p.split('/')[-1].split('.')[0][:12] for p in patients_test]))
        return patients_test

    def get_val_train_patients(self, resample_round):
        patients_train = load_obj('train_img_paths_DX_round_{}'.format(resample_round), self.DATA_SPLIT_DIR + 'train/')
        patients_train = list(set([p.split('/')[-1].split('.')[0][:12] for p in patients_train]))
        patients_val = load_obj('val_img_paths_DX_round_{}'.format(resample_round), self.DATA_SPLIT_DIR + 'val/')
        patients_val = list(set([p.split('/')[-1].split('.')[0][:12] for p in patients_val]))
        return patients_val, patients_train

    def get_test_slides(self):
        patients_test = load_obj('test_img_paths_DX', self.DATA_SPLIT_DIR + 'test/')
        patients_test = list(set([p.split('/')[-1].split('.')[0][:15] for p in patients_test]))
        return patients_test

    def get_val_slides(self, resample_round):
        patients_train = load_obj('train_img_paths_DX_round_{}'.format(resample_round), self.DATA_SPLIT_DIR + 'train/')
        patients_train = list(set([p.split('/')[-1].split('.')[0][:15] for p in patients_train]))
        patients_val = load_obj('val_img_paths_DX_round_{}'.format(resample_round), self.DATA_SPLIT_DIR + 'val/')
        patients_val = list(set([p.split('/')[-1].split('.')[0][:15] for p in patients_val]))
        return patients_val, patients_train


class Conf_Postprocess_Consensus(Conf_Postprocess):
    def __init__(self, c):
        super(Conf_Postprocess_Consensus, self).__init__(c)
        self.PREDS_DIR = self.RES_DIR + 'preds_binarized_consensus/'
        self.OUT_DIR_CARTO = self.OUT_DIR + 'cartography/consensus/'
        self._make_paths()
        self.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND = self.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format('consensus')

        # self.out_DIR_name_format = '{}_round_consensus/'
        # self.out_scores_path_format = self.preds_base_DIR + '{}/' + self.out_DIR_name_format


# Conf_Postprocess_Consensus('MKI67_hi_vs_lo')