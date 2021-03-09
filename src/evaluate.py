from conf import *
from conf_postprocess import *
from preprocess import *
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import spearmanr, pearsonr

'''
Evaluation of predictions against ground truth for both regular and ensemble models.
'''


def get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_in_subset):
    slide_ids_subset = []
    for slide_score_path in all_slide_score_paths:
        slide_id = slide_score_path.split('scores_')[-1].replace('.pkl', '')
        patient_id = slide_id[:c.N_CHAR_PATIENT_ID]
        if patient_id not in patients_in_subset:
            continue
        slide_ids_subset.append(slide_id)
    return slide_ids_subset


class BulkScore:
    def __init__(self, bulk_score_name):
        if bulk_score_name == 'mean_tile_score':
            self.score_fn = self.mean
        elif bulk_score_name == 'percent_positive_tiles':
            self.score_fn = self.percent_positive

    def percent_positive(self, scores, cutoff):
        scores_binarized = [1 if s > cutoff else 0 for s in scores]
        return np.mean(scores_binarized)

    def mean(self, scores, cutoff):
        return np.mean(scores)


class Evaluate:
    def __init__(self, c, preds_folder, slide_ids, patient_ids_to_actual,
                 bulk_score_name, data_split_name):
        self.c = c
        self.preds_folder = preds_folder
        self.slide_ids = slide_ids
        self.patient_ids_to_actual = patient_ids_to_actual
        self.log = Logger(conf_postprocess.OUT_DIR, 'eval_log_{}_{}'.format(data_split_name, preds_folder.split('/')[-2]))
        self.result = {}
        self.best_val_for_final_scoring_method = {'auc': 0, 'auc_cutoff': 0, 'accuracy': 0, 'accuracy_cutoff': 0}
        self.bulk_score_name = bulk_score_name
        self.bulk_score_fn = BulkScore(bulk_score_name).score_fn
        self.cutoff = 0.5

    def _get_tile_scores(self, slide_id):
        tile_scores = load_obj('scores_' + slide_id, self.preds_folder)
        tile_scores = tile_scores[:, 0]
        return tile_scores

    def _calc_slide_score(self, slide_id):
        tile_scores = self._get_tile_scores(slide_id)
        slide_score = self.bulk_score_fn(tile_scores, self.cutoff)
        return slide_score

    def score_slides(self):
        all_scores, all_actuals, all_slide_ids = [], [], []

        for slide_id in self.slide_ids:

            patient_id = slide_id[:self.c.N_CHAR_PATIENT_ID]

            slide_actual = self.patient_ids_to_actual[patient_id]
            slide_score = self._calc_slide_score(slide_id)

            all_actuals.append(slide_actual)
            all_scores.append(slide_score)
            all_slide_ids.append(slide_id)

        unique_patients = list(set([slide_id[:self.c.N_CHAR_PATIENT_ID] for slide_id in all_slide_ids]))
        self.log.print_and_log("n slide IDs scored: {}".format(len(all_scores)))
        self.log.print_and_log("n patient IDs scored: {}".format(len(unique_patients)))

        return all_actuals, all_scores, all_slide_ids

    def evaluate(self, metrics=['correlation']):
        '''
        loads list of slide names for which scores_NAME.pkl, labels_NAME.pkl and names_NAME.pkl exist and uses mean,
        max, etc. to generate several slide level scores.
        Compares resulting scores with all_actuals in provided pkl of all_slide_actuals.pkl, e.g. mapping NAME->%trait etc.
        :return:
        '''

        all_actuals, all_scores, _ = self.score_slides()

        if 'correlation' in metrics:
            spearman_r = spearmanr(all_actuals, all_scores)
            pearson_r = pearsonr(all_actuals, all_scores)
            self.log.print_and_log("Slides scored using: {}".format(self.bulk_score_name))
            self.log.print_and_log("Spearman: (correlation={:.2f}, pvalue={:.2f})".format(spearman_r[0], spearman_r[1]))
            self.log.print_and_log("Pearson:  (correlation={:.2f}, pvalue={:.2f})".format(pearson_r[0], pearson_r[1]))
            self.result['spearman'] = (spearman_r[0], spearman_r[1])
            self.result['pearson'] = (pearson_r[0], pearson_r[1])

        if 'auc' in metrics:
            auc = roc_auc_score(all_actuals, all_scores)
            self.log.print_and_log("AUC using {}: {}".format(self.bulk_score_name, auc))
            self.result['auc'] = auc

        if 'accuracy' in metrics:
            all_scores = np.around(all_scores)  # has to be 0 or 1
            accuracy = accuracy_score(all_actuals, all_scores)
            self.log.print_and_log("Accuracy using {}: {}".format(self.bulk_score_name, accuracy))
            self.result['accuracy'] = accuracy

        print(self.result)


def analyze_predictions(confs, conf_postprocess,
                        resample_round_list=range(5), bulk_scoring_method='percent_positive_tiles', deciding_eval_metric='auc'):
    out_name = '_'.join([c.NAME for c in confs])
    conf_name_to_resample_round_results_auc_test, conf_name_to_resample_round_results_accuracy_test = {}, {}

    for c in confs:

        label_to_numeric_dict = {c.LABELS[0]: 0, c.LABELS[1]: 1}
        patient_to_actual = get_slide_actuals(c.CLINICAL_FILEPATH, 'Patient ID', c.CLINICAL_LABEL_COLS[0],
                                              conf_postprocess.OUT_DIR + 'ground_truth_{}'.format(c.NAME), label_to_numeric_dict)

        resample_round_results_accuracy_val, resample_round_results_auc_val, \
            resample_round_results_spearman_val, resample_round_results_pearson_val = [], [], [], []

        resample_round_results_accuracy_test, resample_round_results_auc_test,\
            resample_round_results_spearman_test, resample_round_results_pearson_test = [], [], [], []

        patients_test = conf_postprocess.get_test_patients()

        for resample_round in resample_round_list:
            preds_folder = conf_postprocess.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format(resample_round)
            all_slide_score_paths = glob.glob(preds_folder + 'scores_*')

            patients_val, patients_train = conf_postprocess.get_val_train_patients(resample_round)

            slide_ids_train = get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_train)
            slide_ids_val = get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_val)
            slide_ids_test = get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_test)

            # Train
            evaluate_train = Evaluate(c, preds_folder, slide_ids_train, patient_to_actual,
                                      bulk_scoring_method, 'train')

            evaluate_train.log.print_and_log(preds_folder)
            evaluate_train.log.print_and_log("\nn train patients: {}".format(len(patients_train)))
            evaluate_train.evaluate(metrics=['correlation', 'auc', 'accuracy'])

            # Validation
            evaluate_val = Evaluate(c, preds_folder, slide_ids_val, patient_to_actual,
                                    bulk_scoring_method, 'val')

            evaluate_val.log.print_and_log(preds_folder)
            evaluate_val.log.print_and_log("\nn val patients: {}".format(len(patients_val)))
            evaluate_val.evaluate(metrics=['correlation', 'auc', 'accuracy'])

            # Test
            evaluate_test = Evaluate(c, preds_folder, slide_ids_test, patient_to_actual,
                                     bulk_scoring_method, 'test')

            evaluate_test.log.print_and_log(preds_folder)
            evaluate_test.log.print_and_log("\nn test patients: {}".format(len(patients_test)))
            evaluate_test.evaluate(metrics=['correlation', 'auc', 'accuracy'])

            # Saving
            resample_round_results_accuracy_val.append(evaluate_val.result['accuracy'])
            resample_round_results_auc_val.append(evaluate_val.result['auc'])
            resample_round_results_spearman_val.append(evaluate_val.result['spearman'])
            resample_round_results_pearson_val.append(evaluate_val.result['pearson'])

            resample_round_results_accuracy_test.append(evaluate_test.result['accuracy'])
            resample_round_results_auc_test.append(evaluate_test.result['auc'])
            resample_round_results_spearman_test.append(evaluate_test.result['spearman'])
            resample_round_results_pearson_test.append(evaluate_test.result['pearson'])

        evaluate_test.log.print_and_log(
            "\n\nSummary stats:\n---------------\n"
            "## Average test accuracy over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                          np.mean(resample_round_results_accuracy_test)))
        evaluate_test.log.print_and_log(
            "## Average test AUC over {} resample rounds: {} ({}, {}) ##".format(len(resample_round_list),
                                                                           np.mean(resample_round_results_auc_test),
                                                                           np.min(resample_round_results_auc_test),
                                                                           np.max(resample_round_results_auc_test)))
        evaluate_test.log.print_and_log(
            "## Average test pearson over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                         np.mean(resample_round_results_pearson_test)))
        evaluate_test.log.print_and_log(
            "## Average test spearman over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                    np.mean(resample_round_results_spearman_test)))

        evaluate_val.log.print_and_log(
            "\n\nSummary stats:\n---------------\n "
            "## Average val accuracy over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                         np.mean(resample_round_results_accuracy_val)))
        evaluate_val.log.print_and_log(
            "## Average val AUC over {} resample rounds: {} ({}, {}) ##".format(len(resample_round_list),
                                                                    np.mean(resample_round_results_auc_val),
                                                                    np.min(resample_round_results_auc_val),
                                                                    np.max(resample_round_results_auc_val)))
        evaluate_val.log.print_and_log(
            "## Average val pearson over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                         np.mean(resample_round_results_pearson_val)))
        evaluate_val.log.print_and_log(
            "## Average val spearman over {} resample rounds: {} ##".format(len(resample_round_list),
                                                                    np.mean(resample_round_results_spearman_val)))

        conf_name_to_resample_round_results_auc_test[c.NAME] = resample_round_results_auc_test
        conf_name_to_resample_round_results_accuracy_test[c.NAME] = resample_round_results_accuracy_test

        save_obj(resample_round_results_accuracy_val, '{}_{}'.format(c.NAME, 'resample_round_results_accuracy_val'), conf_postprocess.OUT_DIR)
        save_obj(resample_round_results_auc_val, '{}_{}'.format(c.NAME, 'resample_round_results_auc_val'), conf_postprocess.OUT_DIR)

        save_obj(resample_round_results_accuracy_test, '{}_{}'.format(c.NAME, 'resample_round_results_accuracy_test'), conf_postprocess.OUT_DIR)
        save_obj(resample_round_results_auc_test, '{}_{}'.format(c.NAME, 'resample_round_results_auc_test'), conf_postprocess.OUT_DIR)

    boxplot_multi(conf_name_to_resample_round_results_auc_test, 'boxplot_{}_auc_{}'.format(out_name, bulk_scoring_method), '', '',
                  y_min=0, y_max=1, out_dir=conf_postprocess.OUT_DIR)


def analyze_test_consensus(c, conf_postprocess,
                           bulk_scoring_method='percent_positive_tiles'):
    conf_postprocess.DATA_SPLIT_DIR = conf_postprocess.DATA_SPLIT_DIR
    label_to_numeric_dict = {c.LABELS[0]: 0, c.LABELS[1]: 1}
    patient_ids_to_actual = get_slide_actuals(c.CLINICAL_FILEPATH, 'Patient ID', c.CLINICAL_LABEL_COLS[0],
                                              '../out/ground_truth_{}'.format(c.NAME), label_to_numeric_dict)
    preds_folder = conf_postprocess.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND

    # patients_test = load_obj('test_img_paths_DX', conf_postprocess.DATA_SPLIT_DIR + 'test/')
    # patients_test = list(set([p.split('/')[-1].split('.')[0][:12] for p in patients_test]))
    patients_test = conf_postprocess.get_test_patients()
    all_slide_score_paths = glob.glob(preds_folder + 'scores_*')
    slide_ids_test = get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_test)

    evaluate = Evaluate(c, preds_folder, slide_ids_test, patient_ids_to_actual,
                        bulk_scoring_method, "test")
    evaluate.log.print_and_log(preds_folder)

    # Test using val best cutoff
    evaluate.log.print_and_log("\n ***************** \nTesting consensus")
    evaluate.log.print_and_log("\nn test patients: {}".format(len(patients_test)))
    evaluate.evaluate(['correlation', 'auc', 'accuracy'])

    test_spearman, test_pearson, test_auc, test_acc = \
        evaluate.result['spearman'], evaluate.result['pearson'], \
        evaluate.result['auc'], evaluate.result['accuracy']

    evaluate.log.print_and_log(
        "## \n Accuracy {}: ##".format(test_acc))
    evaluate.log.print_and_log(
        "## \n AUC {}: ##".format(test_auc))
    evaluate.log.print_and_log(
        "## \n Spearman {} (p-val {}): ##".format(test_spearman[0], test_spearman[1]))

    save_obj(test_acc, '{}_{}'.format(c.NAME, 'consensus_results_accuracy_test'), conf_postprocess.OUT_DIR)
    save_obj(test_auc, '{}_{}'.format(c.NAME, 'consensus_results_auc_test'), conf_postprocess.OUT_DIR)


def scores_for_plots(c, conf_postprocess, resample_round, data, sub_data, correl_type, include_ood_percentiles='all', bulk_scoring_method='percent_positive_tiles'):
    groud_truth_labels = get_slide_actuals(c.CLINICAL_FILEPATH, 'Patient ID', c.CLINICAL_LABEL_COLS[0]+'_50_pctl',
                      '../out/ground_truth_{}'.format(c.NAME+'_50_pctl'), {'hi':1, 'lo':0})
    groud_truth_values = get_slide_actuals(c.CLINICAL_FILEPATH, 'Patient ID', c.CLINICAL_LABEL_COLS[0]+'_50_pctl_value',
                      '../out/ground_truth_{}'.format(c.NAME+'_50_pctl_value'))

    unlabeled_patient_ids = get_ood_patient_ids([c], include_percentiles=include_ood_percentiles)
    unlabeled_patient_ids = [p for p in groud_truth_labels.keys() if p in unlabeled_patient_ids]

    if resample_round != 'consensus':
        # verify unlabeled not in train, val ,test
        patients_test = conf_postprocess.get_test_patients()
        patients_val, patients_train = conf_postprocess.get_val_train_patients(resample_round)
        assert len([p for p in unlabeled_patient_ids if (p in patients_train) or (p in patients_test) or (p in patients_val)]) == 0

    else:
        # verify unlabeled not in train, val ,test using any of the resample_round rounds (resample_round = 0)
        conf_postprocess_tmp = Conf_Postprocess(c)
        patients_test = conf_postprocess_tmp.get_test_patients()
        patients_val, patients_train = conf_postprocess_tmp.get_val_train_patients(resample_round=0)
        assert len([p for p in unlabeled_patient_ids if (p in patients_train) or (p in patients_test) or (p in patients_val)]) == 0

    trait_name = c.NAME.split('_')[0]
    preds_folder = conf_postprocess.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format(resample_round)
    all_slide_score_paths = glob.glob(preds_folder + 'scores_*')
    if sub_data == 'ood' or sub_data == 'ood_near':
        chosen_slides = get_slide_ids_for_patient_subset(c, all_slide_score_paths, unlabeled_patient_ids)
    if sub_data == 'test':
        chosen_slides = get_slide_ids_for_patient_subset(c, all_slide_score_paths, patients_test)
    e = Evaluate(c, preds_folder=preds_folder, slide_ids=chosen_slides,
                 patient_ids_to_actual=groud_truth_labels,
                 bulk_score_name=bulk_scoring_method, data_split_name=sub_data)
    if sub_data == 'ood_near':
        e.log.print_and_log("** Using ood percentiles **: {}".format(include_ood_percentiles))
    else:
        assert sub_data == 'test' or include_ood_percentiles == 'all'
    slide_actuals, slide_scores, slide_ids = e.score_slides()
    patient_ids = [s[:12] for s in slide_ids]
    actuals_values = [groud_truth_values[p] for p in patient_ids]

    save_obj(slide_ids, trait_name+'_{}_slide_ids'.format(sub_data))

    # asserts to remove TODO
    scored_patient_ids = [s[:12] for s in slide_ids]
    if sub_data == 'ood' or sub_data == 'ood_near':
        assert len([p for p in scored_patient_ids if
                    (p in patients_train) or (p in patients_test) or (p in patients_val)]) == 0
        d = pd.read_csv(confs[0].CLINICAL_FILEPATH)
        d[trait_name + 'is_ood_scored'] = np.nan
        for i in range(len(d)):
            if d.loc[i, 'Patient ID'] in scored_patient_ids:
                d.loc[i, trait_name+'is_ood_scored'] = True
        d.to_csv('../res/clinical_verifying_ood.csv', index=False)
    if sub_data == 'test':
        assert len([p for p in scored_patient_ids if
                    (p in patients_train) or (p in patients_val)]) == 0

    e.evaluate(['auc', 'accuracy', 'correlation'])
    scores_in_pos = [slide_scores[i] for i in range(len(slide_scores)) if slide_actuals[i] == 1]
    scores_in_neg = [slide_scores[i] for i in range(len(slide_scores)) if slide_actuals[i] == 0]

    for p in scores_in_pos:
        data['Trait'].append(trait_name)
        data['Slide scores'].append(p)
        data['Ground truth'].append(r'$>$' +' median')
    for p in scores_in_neg:
        data['Trait'].append(trait_name)
        data['Slide scores'].append(p)
        data['Ground truth'].append(r'$\leq$' + ' me'
                                                'dian')

    hue = [r'$>$' +' median' if i > 0.5 else r'$\leq$' + ' median' for i in actuals_values]
    boxplot_multi_from_df(pd.DataFrame.from_dict({'Actual percentiles': actuals_values,
                                                  'Scores' : slide_scores,
                                                  'OOD label': hue}),
                          'Actual percentiles', 'Scores', 'OOD label', '',
                          conf_postprocess.OUT_DIR+'scatter_{}_subdata_{}_resample_round_{}'.format(c.NAME,sub_data,resample_round))

    if correl_type == 'binary':
        correl_data = e.result['spearman']
    else:
        correl_data = spearmanr(actuals_values, slide_scores)
    return data, e.result['auc'], correl_data


def make_consensus_preds(c, n_top, metric='auc'):
    # Generate consensus scores
    model_postprocess_conf = Conf_Postprocess(c)

    top_resample_round_results, top_resample_round_ids = get_top_n_resample_rounds(c, model_postprocess_conf, n_top, metric)

    log.print_and_log("Generating consensus scores")
    first_resample_round = top_resample_round_ids[0]
    first_resample_round_folder = model_postprocess_conf.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format(first_resample_round)
    model_postprocess_conf_consensus = Conf_Postprocess_Consensus(c)
    consensus_preds_path = model_postprocess_conf_consensus.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format('consensus')

    if not os.path.exists(consensus_preds_path):
        os.makedirs(consensus_preds_path)
    files = glob.glob(first_resample_round_folder + 'scores_TCGA-*-*-*.pkl')
    for f in files:
        sample_name = f.split('/')[-1].split('.')[0].split('_')[-1]
        top_resample_round_scores_binary, top_resample_round_scores = [], []
        for resample_round in top_resample_round_ids:
            resample_round_folder = model_postprocess_conf.MODEL_PREDS_DIR_WITH_RESAMPLE_ROUND.format(resample_round)
            scores = load_obj(resample_round_folder + 'scores_{}'.format(sample_name), '')
            if not 'DEBUG' in c.NAME:
                names = load_obj(resample_round_folder + 'names_{}'.format(sample_name), '')
            scores_binary = binarize_using_threshold(scores, 0.5)
            top_resample_round_scores_binary.append(scores_binary)
        consensus_scores = create_consensus_array(top_resample_round_scores_binary).reshape(top_resample_round_scores_binary[0].shape)
        save_obj(consensus_scores, 'scores_{}'.format(sample_name), consensus_preds_path)
        if not 'DEBUG' in c.NAME:
            save_obj(names, 'names_{}'.format(sample_name), consensus_preds_path)


if __name__ == '__main__':

    confs = [Conf_BRCA_TRAITS_miR_17_5p_extreme(), Conf_BRCA_TRAITS_MKI67_extreme()]

    # confs = [Conf_BRCA_TRAITS_miR_17_5p_extreme(), Conf_BRCA_TRAITS_MKI67_extreme(), Conf_BRCA_TRAITS_FOXA1_extreme(),
    #          Conf_BRCA_TRAITS_MYC_extreme(), Conf_BRCA_TRAITS_miR_29a_3p_extreme(),
    #          Conf_BRCA_TRAITS_ESR1_extreme(), Conf_BRCA_TRAITS_CD24_extreme(),Conf_BRCA_TRAITS_FOXC1_extreme(),
    #              Conf_BRCA_TRAITS_ERBB2_extreme(), Conf_BRCA_TRAITS_EGFR_extreme()]

    # confs = [Conf_LUAD_TRAITS_miR_17_5p_extreme(), Conf_LUAD_TRAITS_KRAS_extreme(), Conf_LUAD_TRAITS_CD274_extreme(), Conf_LUAD_TRAITS_miR_21_5p_extreme(), Conf_LUAD_TRAITS_EGFR_extreme()]

    is_consensus = True
    # bulk_scoring_method = 'mean_tile_score'
    bulk_scoring_method = 'percent_positive_tiles'

    if is_consensus:
        resample_round_options = ['consensus']
        conf_postprocess_fn = Conf_Postprocess_Consensus
    else:
        resample_round_options = range(5)
        conf_postprocess_fn = Conf_Postprocess

    for c in confs:

        conf_postprocess = conf_postprocess_fn(c)

        if 'consensus' in conf_postprocess.PREDS_DIR:
            make_consensus_preds(c, n_top=3)
            analyze_test_consensus(c, conf_postprocess, bulk_scoring_method=bulk_scoring_method)
        else:
            log.print_and_log("Running using predictions under: {}".format(conf_postprocess.PREDS_DIR))
            analyze_predictions([c], conf_postprocess, resample_round_list=range(5), bulk_scoring_method=bulk_scoring_method)

    # Boxplot scores and ood analysis consensus
    if is_consensus:
        resample_round = 'consensus'
        # for sub_data in ['test', 'ood', 'ood_near']:
        for sub_data in ['test', 'ood', 'ood_near']:
            for correl_type in ['binary', 'values']:
                if sub_data == 'ood_near':
                    include_ood_percentiles = [0.3, 0.8]  # i.e. 20-30%, 70-80%
                else:
                    include_ood_percentiles = 'all'
                data = {'Trait':[], 'Slide scores':[], 'Ground truth': []}
                pvals, spearman_r = [], []
                for c in confs:
                    conf_postprocess = conf_postprocess_fn(c)
                    data, auc, spearman = scores_for_plots(c, conf_postprocess, resample_round=resample_round, data=data, sub_data=sub_data,
                                                           correl_type=correl_type,
                                                           include_ood_percentiles=include_ood_percentiles,
                                                           bulk_scoring_method=bulk_scoring_method)
                    pvals.append(spearman[1])
                    spearman_r.append(spearman[0])
                label_scores_gt_df = pd.DataFrame.from_dict(data)
                if sub_data == 'ood':
                    title = 'Slide scores vs ground truth for out-of-distribution samples resample round {}'.format(resample_round)
                elif sub_data == 'ood_near':
                    title = 'Slide scores vs ground truth for near-distribution samples resample round {}'.format(resample_round)
                else:
                    title = 'Slide scores vs ground truth for test samples resample round {}'.format(resample_round)
                boxplot_multi_from_df(label_scores_gt_df, 'Trait', 'Slide scores', 'Ground truth', title,
                                      conf_postprocess.OUT_DIR + title)
                pvals_fdr = fdr_correction(pvals)
                pvals = format_numbers(pvals)
                pvals_fdr = format_numbers(pvals_fdr)
                spearman_r = format_numbers(spearman_r)
                pvals_before = list(zip(pvals, [c.NAME for c in confs]))
                pvals_after = list(zip(pvals_fdr, [c.NAME for c in confs]))
                spearmans = list(zip(spearman_r, [c.NAME for c in confs]))
                log = Logger(conf_postprocess.OUT_DIR, 'consensus_pvals')
                log.print_and_log("subdata: {}\n".format(sub_data))
                if sub_data == 'ood_near':
                    log.print_and_log("Using percentiles: {}\n".format(include_ood_percentiles))
                    log.print_and_log("If no. of ood percentiles is 2 then spearman binary and spearman values will"
                                      "be the same.")
                log.print_and_log("correl type: {}\n".format(correl_type))
                log.print_and_log("spearmans: \n{}".format(spearmans))
                log.print_and_log("pvals before fdr correction: \n{}".format(pvals_before))
                log.print_and_log("pvals after fdr correction: \n{}".format(pvals_after))



