from utils import *
from conf import *
from conf_postprocess import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import glob


'''
Functionality for:
(1) Producing tensor molecular cartographies
(2) Producing heterogeneity maps
(3) Computing HTI
(4) Performing survival analysis
'''


log = Logger('../out/', 'log_cartography')


class Cartography():
    def __init__(self, predictions_folder, out_folder):
        '''

        :param predictions_folder: e.g. '../out/tmp/'
        :param out_folder: e.g. '../out/results/'
        '''
        self.out_folder = out_folder
        self.predictions_folder = predictions_folder
        self.slide_results = {}
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    # heatmap
    def heatmap_single_slide(self, slide_id, labels_to_include):
        map_all_labels = None

        for label in labels_to_include:
            label_map = load_obj("{}_{}_map".format(slide_id, label), '{}{}/'.format(self.predictions_folder, label))
            slide_result = np.mean(label_map)
            # slide's avg prediction over its tiles
            self.slide_results[(slide_id, label)] = slide_result
            if map_all_labels is None:
                map_all_labels = label_map
            else:
                map_all_labels = np.append(map_all_labels, label_map, 1)

        save_heatmap(map_all_labels, '{}_combined_heatmap_{}'.format(slide_id, '_'.join(labels_to_include)), self.out_folder)
        save_obj(self.slide_results, 'slide_results_aggregated_by_mean', self.out_folder)
        log.print_and_log("Saved heatmaps and slide-aggregated results to: {}".format(self.out_folder))

    def heatmap_all_slides(self, binarize=False):
        log.print_and_log("Saving to: {}".format(self.out_folder))
        files = glob.glob(self.predictions_folder + 'scores_TCGA-*-*-*.pkl')
        for f in files:
            sample_name = f.split('/')[-1].split('.')[0].split('_')[-1]
            try:
                tile_indices, tile_scores = get_indices_scores_from_dict(self.predictions_folder, 'scores_'+sample_name,
                                                                               'names_'+sample_name)
            except:
                log.print_and_log("Could not find indices_scores dictionary for {}".format(sample_name))
                continue
            map = self.create_map_from_indices_scores(tile_indices, tile_scores)
            save_obj(map, "{}_map".format(sample_name), self.out_folder)
            if binarize:
                map = np.around(map)
            save_heatmap(map, sample_name+'.png', self.out_folder, legend=False)

    def create_map_from_indices_scores(self, indices, scores):
        '''

        :param indices: e.g.: ((0,1), (0,2),...,(1,0),(1,1)..)
        :param scores:  corresp. float for each cell
        :param base_shape: matrix size of map
        :return:
        '''

        base_shape = np.max(indices,0) + [1, 1]
        map = np.zeros(base_shape)
        for k in range(len(scores)):
            (i, j) = indices[k]
            score = scores[k]
            map[i, j] = score
        if len(map) == 0:
            map = np.zeros((1,1))
        map = np.transpose(map)
        return map


if __name__ == '__main__':

    confs = [Conf_BRCA_TRAITS_MKI67_extreme()]
    # confs = [Conf_BRCA_TRAITS_miR_17_5p_extreme()]
    # confs = [Conf_BRCA_TRAITS_FOXA1_extreme()]
    # confs = [Conf_BRCA_TRAITS_MYC_extreme()]
    # confs = [Conf_BRCA_TRAITS_miR_29a_3p_extreme()]
    # confs = [Conf_BRCA_TRAITS_FOXC1_extreme()]
    # confs = [Conf_BRCA_TRAITS_EGFR_extreme()]
    # confs = [Conf_BRCA_TRAITS_CD24_extreme()]
    # confs = [Conf_BRCA_TRAITS_ESR1_extreme()]
    # confs = [Conf_BRCA_TRAITS_ERBB2_extreme()]
    # #
    # confs = [Conf_LUAD_TRAITS_KRAS_extreme()]
    # confs = [Conf_LUAD_TRAITS_EGFR_extreme()]
    # confs = [Conf_LUAD_TRAITS_CD274_extreme()]
    # confs = [Conf_LUAD_TRAITS_miR_17_5p_extreme()]
    # confs = [Conf_LUAD_TRAITS_miR_21_5p_extreme()]

    # confs = [Conf_DEBUG()]


    make_top_resample_round_consensus = True
    generate_cartographies = True
    metric = 'auc'
    n_top = 3


    # confs = [Conf_BRCA_TRAITS_T_stage_extremes(), Conf_BRCA_TRAITS_MKI67_extreme(), Conf_BRCA_TRAITS_ERBB2_extreme(), Conf_BRCA_TRAITS_ESR1_extreme(), Conf_BRCA_TRAITS_FRACTION_ALTERED(), ]
    # confs = [Conf_BRCA_TRAITS_MYC_extreme(), Conf_BRCA_TRAITS_FOXA1_extreme(), Conf_BRCA_TRAITS_EGFR_extreme(), Conf_BRCA_TRAITS_CD24_extreme(), Conf_BRCA_TRAITS_ER(), Conf_BRCA_TRAITS_FOXC1_extreme(), Conf_BRCA_TRAITS_KRT14_extreme()]

    for c in confs:
        n_resample_round = c.N_ROUNDS
        model_postprocess_conf = Conf_Postprocess(c)
        preds_model_folder = model_postprocess_conf.MODEL_PREDS_DIR
        data_split_model_folder = model_postprocess_conf.DATA_SPLIT_DIR

        patients_test = model_postprocess_conf.get_test_patients()#load_obj('test_patient_ids', data_split_model_folder + 'test/')

        top_resample_round_results, top_resample_round_ids = get_top_n_resample_rounds(c, model_postprocess_conf, n_top, metric)

        if make_top_resample_round_consensus:

            # Generate consensus scores
            log.print_and_log("Generating consensus scores")
            first_resample_round = top_resample_round_ids[0]
            first_resample_round_folder = model_postprocess_conf.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format(first_resample_round)
            model_postprocess_conf_consensus = Conf_Postprocess_Consensus(c)
            consensus_preds_path = model_postprocess_conf_consensus.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format('consensus')

            # Generate heatmap images and numpy pkl
            if not 'DEBUG' in c.NAME and generate_cartographies:
                log.print_and_log("Generating consensus cartography heatmaps.")
                cartography_out_path = model_postprocess_conf_consensus.OUT_DIR_CARTO + model_postprocess_conf_consensus.MODEL_DIR
                cartography = Cartography(consensus_preds_path, cartography_out_path)
                cartography.heatmap_all_slides(binarize=True)