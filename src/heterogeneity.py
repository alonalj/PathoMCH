from conf_postprocess import Conf_Postprocess, Conf_Postprocess_Consensus
from utils import *
from conf import *
import glob


'''
Functionality for:
(1) Producing tensor molecular cartographies
(2) Producing heterogeneity maps
(3) Computing HTI
(4) Performing survival analysis
'''


def get_score_maps(confs, sample_name):
    '''
    Reads confs' score maps and returns binary tensor maps
    :param confs:
    :param sample_name:
    :return:
    '''
    maps = []
    for c in confs:
        folder_name = c.NAME
        filename_format = PATH_TO_MAPS + folder_name + '/' + sample_name +'_map'
        map = load_obj(filename_format, '')
        maps.append(map)
    maps = np.around(maps)
    return maps


def create_heterogeneity_map(confs, sample_name, out_folder, score):
    '''
    Generates heterogeneity maps for the combined traits included in confs (a list of Conf objects, one per trait. See
    example in conf.py)
    :return:
    '''
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    assert len(confs) == 2, "Currently supports only two traits"
    maps = get_score_maps(confs, sample_name)
    maps = np.around(maps)
    d_sum = sum([2 ** i * maps[i] for i in range(len(maps))])
    all_names = [c.NAME.split('_')[0] for c in confs]
    if 'luad' in confs[0].TCGA_COHORT_NAME:
        colors = [(255, 255, 255), (177 , 215, 235), (198, 164, 147)]
    elif 'brca' in confs[0].TCGA_COHORT_NAME:
        colors = [(255, 255, 255), (240, 174, 205), (203, 235, 170)]
    # adding mean between first two non-whites
    colors.append(((colors[1][0]+colors[2][0]) / 2, (colors[1][1]+colors[2][1]) / 2, (colors[1][2]+colors[2][2]) / 2))
    colors = [(c[0]/255., c[1]/255., c[2]/255.) for c in colors]
    all_name_combinations = powerset(all_names)
    combination_color_id = np.arange(0, len(all_name_combinations)) / (len(all_name_combinations)-1)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cmap = ListedColormap(colors)
    color_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(combination_color_id))]
    color_names = ['Neither', all_name_combinations[1][0], all_name_combinations[2][0], 'Both']
    save_heatmap(d_sum, '{}_{}_test_hetero_score_{:.2f}.png'.format(sample_name, 'heterogeneity', score), out_folder, vmax=np.max(d_sum),
                 cmap=cmap, legend={'colors': color_lines, 'names': color_names}, reverse_color=True)


def HTI(tensor_mol_cart):
    '''
    Computes the heterogeneity index (HTI). 1 = heterogeneous, 0 = homogeneous.

    data is a tensor molecular cartography, where each layer corresponds to a single trait. Expects a numpy array
    of shape: N_LOCATIONS x N_TRAITS containing binary values.
    Example:
    data = np.array([[0,1], [1,1], [1,0]]) represents three locations and two traits. The first location only has the
    second trait, the second location has both traits, and the third location has only the first trait. HTI will be 1.
    :return: HTI
    '''
    from scipy.stats import entropy
    assert tensor_mol_cart.shape[-1] == 2, "Currently supports two molecular traits."
    probs = []
    tensor_mol_cart = tensor_mol_cart.astype(int)
    a, b = tensor_mol_cart[:, 0], tensor_mol_cart[:, 1]
    only_a = [1 for i in range(len(a)) if a[i] == 1 and b[i] == 0]
    only_b = [1 for i in range(len(b)) if b[i] == 1 and a[i] == 0]
    a_and_b = a & b
    where_any_positive = (a | b) == 1
    for type in [only_a, only_b, a_and_b]:
        if np.count_nonzero(where_any_positive) == 0:
            # all positions are empty - trivially homogeneous
            return 0
        else:
            p = np.count_nonzero(type) / np.count_nonzero(where_any_positive)
        probs.append(p)

    entr = entropy(probs, base=3)

    return entr


def make_tensor_cartography(confs, sample_name):
    '''
    Create tensor cartography for traits in confs.
    '''
    maps = get_score_maps(confs, sample_name)
    maps = [np.expand_dims(m, -1) for m in maps]
    tensor_map = np.concatenate(maps, -1)
    tensor_map = np.reshape(tensor_map, (tensor_map.shape[0] * tensor_map.shape[1], -1))
    # remove rows (pixels) with no positive (1) predictions at all:
    tensor_map = tensor_map[~np.all(tensor_map == 0, axis=1)]
    return tensor_map


def add_heterogeneity_scores_to_clinical_data(heterogeneity_score_col_name, heterogeneity_scores, clinical_data_path):
    d = pd.read_csv(clinical_data_path)
    d[heterogeneity_score_col_name] = None
    for i in range(len(d)):
        patient_id = d.loc[i, 'Patient ID']
        if patient_id in heterogeneity_scores.keys():
            d.loc[i, heterogeneity_score_col_name] = heterogeneity_scores[patient_id]
    out_path = clinical_data_path
    d.to_csv(out_path, index=False)


def add_baseline_score_to_clinical_data(confs, baseline_name, heterogeneity_score_col, clinical_data_path):
    baseline_score_col = heterogeneity_score_col + '_baseline_{}'.format(baseline_name)
    d = pd.read_csv(clinical_data_path)
    if 'both_hi_lo' == baseline_name:
        cols = [confs[0].CLINICAL_LABEL_COLS[0] + '_50_pctl', confs[1].CLINICAL_LABEL_COLS[0] + '_50_pctl']
        pos = np.where((d[cols[0]] == 'hi') & (d[cols[1]] == 'hi'))[0].tolist()
        neg = np.where((d[cols[0]] == 'lo') & (d[cols[1]] == 'lo'))[0].tolist()
    if 'first_hi_lo' == baseline_name:
        cols = [confs[0].CLINICAL_LABEL_COLS[0] + '_50_pctl']
        pos = np.where(d[cols[0]] == 'hi')[0].tolist()
        neg = np.where(d[cols[0]] == 'lo')[0].tolist()
    if 'second_hi_lo' == baseline_name:
        cols = [confs[1].CLINICAL_LABEL_COLS[0] + '_50_pctl']
        pos = np.where(d[cols[0]] == 'hi')[0].tolist()
        neg = np.where(d[cols[0]] == 'lo')[0].tolist()
    if 'one_hi_other_lo' == baseline_name:
        cols = [confs[0].CLINICAL_LABEL_COLS[0] + '_50_pctl', confs[1].CLINICAL_LABEL_COLS[0] + '_50_pctl']
        pos = np.where(((d[cols[0]] == 'hi') & (d[cols[1]] == 'lo')) | ((d[cols[0]] == 'lo') & (d[cols[1]] == 'hi')))[0].tolist()
        neg = np.where(((d[cols[0]] == 'hi') & (d[cols[1]] == 'hi')) | ((d[cols[0]] == 'lo') & (d[cols[1]] == 'lo')))[0].tolist()
    d[baseline_score_col] = np.nan
    d.loc[pos, baseline_score_col] = 1
    d.loc[neg, baseline_score_col] = 0
    out_path = clinical_data_path
    d.to_csv(out_path, index=False)


def plot_score_distribution_per_type(d, conf_postprocess, heterogeneity_col, category_col, y_label):
    name = 'heterogeneity_distribution_per_type_{}_{}'.format(heterogeneity_col, category_col)
    d = d.dropna(subset=[heterogeneity_col, category_col])
    print("num participating in boxplot", len(d))
    boxplot_simple(d, category_col, heterogeneity_col, y_label, conf_postprocess.OUT_DIR+name)


def KM(confs, conf_postprocess, lo_cutoff=0.5, hi_cutoff=0.5, time_column="Overall Survival (Months)",
       status_column="Overall Survival Status", status_negative="DECEASED", subdata='test_only',
       is_baseline=False, baseline_name=None):
    from lifelines import KaplanMeierFitter

    run_name = '_'.join([c.NAME for c in confs])
    heterogeneity_score_col = 'heterogeneity_score_{}'.format(run_name)
    heterogeneity_score_col_baseline = 'heterogeneity_score_{}_baseline_{}'.format(run_name, baseline_name)
    assert len(confs) == 2, "Currently supports only two molecular traits."

    clinical_data_km = pd.read_csv(confs[0].CLINICAL_FILEPATH)

    if (not is_baseline) or use_same_patients_for_baseline_and_heterogeneity:
        if subdata == 'ood_or_test':
            # test and OOD patients in intersection of confs OODs
            patients_test = load_obj('test_patient_ids', conf_postprocess.DATA_SPLIT_DIR + 'test/')
            ood_patients = get_ood_patient_ids(confs, include_percentiles=include_ood_percentiles)
            patients_to_keep = ood_patients
            patients_to_keep.extend(patients_test)
            patients_to_keep = set(patients_to_keep)
            patients_ix = clinical_data_km['Patient ID'].isin(patients_to_keep)
            clinical_data_km = clinical_data_km[patients_ix]
        if subdata == 'ood':
            ood_patients = get_ood_patient_ids(confs, include_percentiles=include_ood_percentiles)
            patients_to_keep = ood_patients
            patients_ix = clinical_data_km['Patient ID'].isin(patients_to_keep)
            clinical_data_km = clinical_data_km[patients_ix]

        in_between_ix = clinical_data_km[heterogeneity_score_col].between(lo_cutoff, hi_cutoff)
        clinical_data_km = clinical_data_km[~in_between_ix]

    simple_trait_names = [c.NAME.split('_')[0] for c in confs]
    if not is_baseline and confs[0].TCGA_COHORT_NAME == 'brca':
        y_label = 'HT-Index for {} and {}'.format(simple_trait_names[0], simple_trait_names[1])
        plot_score_distribution_per_type(clinical_data_km, conf_postprocess, heterogeneity_score_col, "PAM50", y_label)
        plot_score_distribution_per_type(clinical_data_km, conf_postprocess, heterogeneity_score_col, 'ER Status By IHC', y_label)

    if use_same_patients_for_baseline_and_heterogeneity:
        clinical_data_km = clinical_data_km.dropna(subset=[time_column, status_column, heterogeneity_score_col ,heterogeneity_score_col_baseline])
    else:
        if is_baseline:
            heterogeneity_score_col = heterogeneity_score_col_baseline
        clinical_data_km = clinical_data_km.dropna(subset=[time_column, status_column, heterogeneity_score_col])

    if is_baseline:
        heterogeneity_score_col = heterogeneity_score_col_baseline

    groups = clinical_data_km[heterogeneity_score_col]
    ix = (groups > 0.5)

    time = clinical_data_km[time_column]  # could use: 'Progress Free Survival (Months)' or "Overall Survival (Months)"
    time = time / 12
    timelim = 10
    time[time > timelim] = timelim
    status = clinical_data_km[status_column] == status_negative
    status[time >= timelim] = False

    if is_baseline:
        if baseline_name == 'both_hi_lo':
            neg_label = 'Both '+ r'$\leq$'+' median'
            pos_label = 'Both '+ r'$>$'+' median'
        if baseline_name == 'first_hi_lo':
            neg_label = simple_trait_names[0] + r' $\leq$'+' median'
            pos_label = simple_trait_names[0] + r' $>$'+' median'
        if baseline_name == 'second_hi_lo':
            neg_label = simple_trait_names[1] + r' $\leq$'+' median'
            pos_label = simple_trait_names[1] + r' $>$'+' median'
        if baseline_name == 'one_hi_other_lo':
            neg_label = 'Same side of median'
            pos_label = 'Opposite sides of median'
        n_info = "n {}: {} \nn {} : {}" \
            .format(pos_label.lower(), len(clinical_data_km[ix]), neg_label.lower(), len(clinical_data_km[~ix]))
    else:
        neg_label = 'HT-Index ' + r'$\leq$'+' {}'.format(lo_cutoff)
        pos_label = 'HT-Index ' + r'$>$' + ' {}'.format(hi_cutoff)
        n_info = ("n" + r'$>$' + "{}: {}" + "\nn" + r'$\leq$' + '{}: {}') \
            .format(hi_cutoff, len(clinical_data_km[ix]), lo_cutoff, len(clinical_data_km[~ix]))

    from lifelines.statistics import logrank_test
    T1, E1 = time[ix], status[ix]
    T2, E2 = time[~ix], status[~ix]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    results.print_summary()
    p = results.p_value

    # plotting
    ax = plt.subplot(111)
    if len(time[ix]) ==0 or len(time[~ix]) == 0:
        print("Empty time data")
        return

    kmf_pos = KaplanMeierFitter()
    ax = kmf_pos.fit(time[ix], status[ix], label=pos_label).plot(ax=ax)

    kmf_neg = KaplanMeierFitter()
    ax = kmf_neg.fit(time[~ix], status[~ix], label=neg_label).plot(ax=ax)

    if p > 1:
        plt.close()
        return p
    if p < 0.01:
        ax.text(0.3, 0.2, 'p-val: {:.0e}'.format(p))
    else:
        ax.text(0.3, 0.2, 'p-val: {:.2f}'.format(p))
    ax.text(0.3, 0.3, n_info)
    ax.grid(False)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time (years)")
    plt.ylabel("Percent survival")

    if use_same_patients_for_baseline_and_heterogeneity:
        patient_basis = baseline_name
    else:
        patient_basis = None
    name_format = '{}/km_{}_time_{}_subdata_{}_cutoffs_{}_{}_is_baseline_{}_patient_basis_{}_include_ood_percentiles_{}.tiff'
    plt.savefig(name_format.format(conf_postprocess.OUT_DIR, heterogeneity_score_col, time_column,
                                   subdata, lo_cutoff, hi_cutoff, is_baseline, patient_basis, include_ood_percentiles),
                dpi=700)
    plt.close()
    return p


if __name__ == '__main__':

    # settings
    confs = [Conf_BRCA_TRAITS_MKI67_extreme(), Conf_BRCA_TRAITS_miR_17_5p_extreme()]
    # confs = [Conf_LUAD_TRAITS_KRAS_extreme(), Conf_LUAD_TRAITS_miR_17_5p_extreme()]

    is_baseline = False
    # baseline_name = 'second_hi_lo'
    # baseline_name = 'first_hi_lo'
    # baseline_name = 'both_hi_lo'
    baseline_name = 'one_hi_other_lo'
    use_same_patients_for_baseline_and_heterogeneity = False
    score_lo_cutoff, score_hi_cutoff = 0.5, 0.5
    # include_ood_percentiles = [0.3,0.8]
    include_ood_percentiles = 'all'
    subdata = 'ood_or_test'

    conf_postprocess = Conf_Postprocess(confs[0])  # any conf id from the cohort will do. Take first.
    conf_postprocess_consensus = Conf_Postprocess_Consensus(confs[0])
    PATH_TO_MAPS = conf_postprocess_consensus.OUT_DIR_CARTO
    out_folder = conf_postprocess.OUT_DIR + 'heterogeneity/'

    confs_combination = [confs]

    for confs in confs_combination:
        print("Running {}".format(confs))
        run_name = '_'.join([c.NAME for c in confs])
        heterogeneity_score_col = 'heterogeneity_score_{}'.format(run_name)

        add_baseline_score_to_clinical_data(confs, baseline_name, heterogeneity_score_col, clinical_data_path=confs[0].CLINICAL_FILEPATH)

        patient_id_to_heterogeneity_score = {}
        for map_path in glob.glob(PATH_TO_MAPS + confs[0].NAME + '/*map.pkl'):
            sample_name = get_minimal_slide_identifier(map_path)
            if 'DX1' not in sample_name:
                continue
            patient_id = sample_name[:confs[0].N_CHAR_PATIENT_ID]
            tensor_molecular_cartography = make_tensor_cartography(confs, sample_name)
            heterogeneity_score = HTI(tensor_molecular_cartography)
            if not heterogeneity_score:
                continue

            if len(confs) == 2:
                # generate heterogeneity maps for test patients that are in-distribution for at least one trait
                test_patients_a = Conf_Postprocess(confs[0]).get_test_patients()
                test_patients_b = Conf_Postprocess(confs[1]).get_test_patients()
                if patient_id in test_patients_a or patient_id in test_patients_b:
                    create_heterogeneity_map(confs, sample_name, out_folder + run_name + '/', heterogeneity_score)

            patient_id_to_heterogeneity_score[patient_id] = heterogeneity_score

        add_heterogeneity_scores_to_clinical_data(heterogeneity_score_col_name=heterogeneity_score_col,
                                                  heterogeneity_scores=patient_id_to_heterogeneity_score,
                                                  clinical_data_path='../res/clinical_data_{}.csv'
                                                  .format(confs[0].TCGA_COHORT_NAME))

        p = KM(confs, conf_postprocess, score_lo_cutoff, score_hi_cutoff,
               time_column="Overall Survival (Months)", status_column="Overall Survival Status",
               status_negative="DECEASED", subdata=subdata, is_baseline=is_baseline, baseline_name=baseline_name)
