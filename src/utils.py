import tensorflow as tf
import pickle as pkl
import numpy as np
import os
from PIL import Image
import pandas as pd
import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt


class Logger:
    '''
    Save info to a text file for later inspection.
    '''
    def __init__(self, folder, name='default'):
        log_file = os.path.join(folder, 'log_{}.txt'.format(name))
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.log_writer = open(log_file, "a")
        self.log_writer.write("\n")
        self.log_writer.flush()

    def print_and_log(self, msg):
        print("\n"+msg)
        self.log_writer.write("\n"+msg)
        self.log_writer.flush()


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def save_obj(obj, name, directory='../res'):
    with open(directory + '/' + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def load_obj(name, directory='../res/'):
    with open(directory + name + '.pkl', 'rb') as f:
        return pkl.load(f)


def get_minimal_slide_identifier(slide_string):
    '''

    :param slide_string: any string that contains the slide name. Assumes TCGA slide name format.
    :return:
    '''
    if '_' in slide_string:
        return slide_string.split('/')[-1].split('_')[0]
    else:
        return slide_string.split('/')[-1].split('.')[0]


def create_reversed_dictionary(dict_to_reverse, reversed_key_maps_to_unique_value):
    '''
    reverses dictionary. can handle a dictionary with value as list or np array as well.
    :param dict_to_reverse:
    :param reversed_key_maps_to_unique_value: True if the current dictionary's value is unique (e.g. IDs), False if not (e.g. labels)
    :return:
    '''
    reversed_dict = {}
    if reversed_key_maps_to_unique_value:  # normally will be false for map: {img_path : label}
        for key, val in dict_to_reverse.items():
            new_key, new_val = val, key
            if type(new_key) == list or type(new_key) == np.ndarray:
                for nk in new_key:
                    reversed_dict[nk] = new_val
            else:
                reversed_dict[nk] = new_val
    else:
        for key, val in dict_to_reverse.items():
            new_key, new_val = val, key
            if type(new_key) == list or type(new_key) == np.ndarray:
                for nk in new_key:
                    if nk in reversed_dict.keys():
                        if new_val not in reversed_dict[nk]:
                            reversed_dict[nk].append(new_val)
                    else:
                        reversed_dict[nk] = [new_val]

            else:
                if new_key in reversed_dict.keys():
                    reversed_dict[new_key].append(new_val)
                else:
                    reversed_dict[new_key] = [new_val]
            # keep only unique values

    return reversed_dict


def boxplot_multi(dict, title, x_label, y_label, y_min=None, y_max=None, out_dir='../out/'):
    '''
    Example:
    e.g. dict = {'x1 actual': [y_1 pred for tile 1, y_2 pred for tile 2, ...], 'x2 actual': [y_2 preds for tile 1, ..]}
    '''

    keys = sorted(dict.keys())
    result = []
    for key in keys:
        result.append(dict[key])

    fontdictx = {'fontsize': 10,
                 'horizontalalignment': 'center'}

    fontdicty = {'fontsize': 10,
                 'verticalalignment': 'baseline',
                 'horizontalalignment': 'center'}
    fig, ax = plt.subplots()
    ax.boxplot(result, showfliers=False)
    ax.set_xticklabels(keys, rotation=90, fontdict={'fontsize': 8})
    if y_min and y_max:
        ax.set(ylim=(y_min, y_max))

    plt.xlabel(x_label, fontdictx)
    plt.ylabel(y_label, fontdicty)
    plt.yticks(fontsize=8)#, rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir + '{}.png'.format(title))
    plt.close()


def boxplot_multi_from_df(data, x_col, y_col, hue_col, title, out_path):
    '''
    data: a pandas df with 3 columns: x_col containing the category, y_col containing the value,
            hue_col containing the category determining the hue.
            Example: df={'Gender':['F', 'F', 'M'], 'Height': [1.62, 1.7, 1.9], 'Smoker':['Y','N','Y']}.
            Height would be y values (x_col=df['Gender'], y_col=df['Height'], hue_col=df['Smoker'])
    '''
    fontdictx = {'fontsize': 12,
                 'horizontalalignment': 'center'}

    fontdicty = {'fontsize': 12,
                 'verticalalignment': 'baseline',
                 'horizontalalignment': 'center'}
    if 'luad' in out_path:
        palette = {r'$>$' +' median': "#93B4C6", '$\\leq$ median': "#63686E"}  #c6a493
        width = 0.3
    elif 'brca' in out_path:
        palette = {r'$>$' +' median': "#D19EB5", '$\\leq$ median': "#63686E"}  #b5d19e
        width = 0.6
    else:
        palette = "Set3" # bright yellow and green
        width = 0.3
    fig, ax = plt.subplots()
    ax = sns.boxplot(x=x_col, y=y_col, hue=hue_col,
                     data=data,
                     palette=palette,
                     medianprops={'color':'white'},
                     width=width,
                     showcaps=False,
                     whiskerprops=dict(linewidth=0.3, color='black'),
                     flierprops=dict(linewidth=0.3, markeredgewidth=0.3, marker='o', markersize=3, color='black'),
                     boxprops=dict(linewidth=0.)
                     )
    plt.xlabel(x_col, fontdictx,  fontsize=12)
    plt.ylabel(y_col, fontdicty,  fontsize=12)
    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], color=palette[r'$>$' +' median'], lw=8, label=r'$>$' +' median'),
                       Line2D([0], [0], color=palette['$\\leq$ median'], lw=8, label='$\\leq$ median')]

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tick_params('both', labelsize='10')
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., title=hue_col, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path+'.tiff', dpi=800)
    plt.close()


def boxplot_simple(df, x_col, y_col, y_label=None, out_path='../out/', palette="Purples"):
    sns.boxplot(x=df[x_col], y=df[y_col], palette=palette)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path+'.tiff', dpi=100)
    plt.close()


def dict_to_str(dict):
    return '_'.join("{!s}_{!r}".format(key, val) for (key, val) in dict.items())


def get_pandas_df_where_col_contains(d, col_name, str_contained):
    return d.loc[d[col_name].str.contains(str_contained), :]


def get_pandas_df_where_col_not_contains(d, col_name, str_not_contained):
    return d.loc[~ d[col_name].str.contains(str_not_contained), :]


def create_list_to_remove_from_thumbnail_folder(to_remove_folder_path):
    to_remove = os.listdir(to_remove_folder_path)
    to_remove_lst = [p.split('_')[0]+'.svs' for p in to_remove]
    save_obj(to_remove_lst, 'slides_to_remove_lung')
    print(to_remove_lst)
    return to_remove_lst


def powerset(iterable):
    from itertools import chain, combinations
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return list(chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1)))


def save_heatmap(matrix, name, out_folder, reverse_color=False, vmin=0, vmax=1, cmap=None, legend=None):
    if not cmap:
        if reverse_color:
            cmap = sns.cm.rocket_r
        else:
            cmap = sns.cm.rocket
    if legend and legend['colors']:
        ax = sns.heatmap(matrix, linewidth=0, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False, yticklabels=False, xticklabels=False)
        ax.legend(legend['colors'], legend['names'], bbox_to_anchor=(-0.09, 0.8), loc=2, borderaxespad=0, frameon=False)
    elif legend is False:
        ax = sns.heatmap(matrix, linewidth=0, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False, cbar=False)
    else:
        ax = sns.heatmap(matrix, linewidth=0, cmap=cmap, vmin=vmin, vmax=vmax, yticklabels=False, xticklabels=False)
    plt.savefig('{}{}'.format(out_folder,name))#, dpi=600)
    plt.close()


def count_examples_in_tfrecord(tf_records_filenames, sub_data_name, folder_path):
    try:
        c = load_obj('count_examples_{}'.format(sub_data_name), folder_path)
        print("loaded count")
        return c
    except:
        c = 0
        for fn in tf_records_filenames:
            for record in tf.python_io.tf_record_iterator(fn):
                c += 1
        save_obj(c, 'count_examples_{}'.format(sub_data_name), folder_path)
    return c


def remove_slides_from_manifest(manifest_path, slides_list):
    slides_list = list(slides_list)
    m = pd.read_csv(manifest_path, sep='\t')
    len_before_removal = len(m)
    m = m.loc[-m['filename'].isin(slides_list), :]
    m.to_csv(manifest_path.split('.txt')[0]+'_AFTER_removal.txt', index=False, sep='\t')
    return m


def create_patient_ids_list(filename, sample_id_column_name, out_name, patient_ids_to_remove=None):
    d = pd.read_csv('../res/{}'.format(filename))
    patient_ids = list(d[sample_id_column_name])
    previous_len_patient_ids = len(patient_ids)
    print("File has {} patient ids".format(previous_len_patient_ids))
    if patient_ids_to_remove:
        print("Removing {} patient ids".format(len(patient_ids_to_remove)))
        patient_ids_to_remove = set(patient_ids_to_remove)
        patient_ids = [p for p in patient_ids if p not in patient_ids_to_remove]
    print("Final list has {} patient ids".format(len(patient_ids)))
    save_obj(patient_ids, out_name)
    print("Saved as {}".format(out_name))
    return d


def get_slide_actuals(filepath, sample_id_column_name, ground_truth_col_name, out_filepath, label_to_numeric_dict=None):
    d = pd.read_csv(filepath)
    d = d[[sample_id_column_name, ground_truth_col_name]]
    d = d.dropna()
    patient_id_to_gt = {}
    patient_ids = list(d[sample_id_column_name])
    ground_truth = list(d[ground_truth_col_name])
    if label_to_numeric_dict:
        ground_truth = [label_to_numeric_dict[l] for l in ground_truth]
    for i in range(len(patient_ids)):
        patient_id_to_gt[patient_ids[i]] = float(ground_truth[i])
    print(patient_id_to_gt)
    save_obj(patient_id_to_gt, out_filepath)
    return patient_id_to_gt


def get_indices_scores_from_dict(dir, scores_dictionary, names_dictionary):
    tile_scores = load_obj(scores_dictionary, dir)
    tile_names = load_obj(names_dictionary,dir)
    tile_names = [t.decode('utf-8').split('.')[1].split('_')[1:] for t in tile_names]
    tile_names = np.array(tile_names).astype(np.int)
    return tile_names, tile_scores


def create_consensus_array(list_of_arrays):
    from scipy.stats import mode
    list_of_arrays_expanded = []
    for m in list_of_arrays:
        m = np.expand_dims(m, -1)
        list_of_arrays_expanded.append(m)
    concat_maps = np.concatenate(list_of_arrays_expanded, -1)
    consensus_map = mode(concat_maps, -1)[0]
    return consensus_map


def binarize_using_threshold(a, cutoff):
    return (a > cutoff) * 1


def get_ood_patient_ids(confs, include_percentiles='all'):
    '''
    Returns list of patient ids for which the label columns used to develop the models of all confs
    is nan (if one conf has a label for that patient the patient is not included)
    :param confs:
    ;:param include_percentiles: which residual expression percentiles to include (e.g. [0.3, 0.4, 0.8] will
    include expression levels between (0.2-0.4] and (0.7-0.8]
    :return:
    '''
    clinical_data = pd.read_csv(confs[0].CLINICAL_FILEPATH)
    label_cols = [c.CLINICAL_LABEL_COLS[0] for c in confs]
    if include_percentiles != 'all':
        for col in label_cols:
            values = clinical_data[col + '_50_pctl_value']
            ix_include = [True if v in include_percentiles else False for v in values]
            clinical_data = clinical_data.loc[ix_include]
    # checking if there's a label per cell for the columns used for training:
    is_cell_na = clinical_data[label_cols].isna()
    # converting to whether all labels per row (patient) are nan (patient was not in model development in both traits)
    is_row_na = is_cell_na.all(axis='columns')
    n_nan = sum(is_row_na*1)
    unlabeled_patient_ids = clinical_data[is_row_na]['Patient ID']
    return list(unlabeled_patient_ids)


def get_labeled_patient_ids(confs):
    '''
    Returns list of patient ids for which the label columns used to develop the models of all confs
    is NOT nan (if one conf has a missing label for that patient the patient is not included)
    :param confs:
    :return:
    '''
    clinical_data = pd.read_csv(confs[0].CLINICAL_FILEPATH)
    label_cols = [c.CLINICAL_LABEL_COLS[0] for c in confs]
    # checking if there's a label per cell:
    is_cell_na = clinical_data[label_cols].isna()
    # converting to whether there's a label per row (patient)
    is_row_na = is_cell_na.all(axis='columns')
    n_nan = sum(is_row_na*1)
    labeled_patient_ids = clinical_data[~is_row_na]['Patient ID']
    assert len(labeled_patient_ids) == n_nan
    return list(labeled_patient_ids)


def fdr_correction(pvals):
    from statsmodels.stats.multitest import multipletests
    _, pvals_fdr_corrected, _, _ = multipletests(pvals, method='fdr_bh', is_sorted=False, returnsorted=False)
    return pvals_fdr_corrected


def get_top_n_resample_rounds(c, conf_postprocess, n, metric):
    resample_round_results = load_obj('{}_{}'.format(c.NAME, 'resample_round_results_{}_val'.format(metric)), conf_postprocess.OUT_DIR)
    zipped = list(zip(resample_round_results, range(len(resample_round_results))))  # range because results are saved in order from 0 to 4
    sorted_hi_lo_auc = sorted(zipped, key=lambda x: x[0], reverse=True)
    top_resample_rounds_data = sorted_hi_lo_auc[:n]
    top_resample_round_results, top_resample_round_ids = [i[0] for i in top_resample_rounds_data], [i[1] for i in top_resample_rounds_data]
    return top_resample_round_results, top_resample_round_ids


def read_pam50_type(c):
    pam50 = pd.read_csv('../res/sampleinfo_TCGA_nanodissect.txt', sep='\t')
    pam50.rename(columns={'submitted_donor_id' : 'Patient ID'}, inplace=True)
    patient_to_pam50 = {}
    for i in range(len(pam50['Patient ID'])):
        p = pam50['Patient ID'][i]
        l = pam50['PAM50'][i]
        patient_to_pam50[p] = l
    clinical = pd.read_csv(c.CLINICAL_FILEPATH)
    clinical['PAM50'] = None
    for i in range(len(clinical['Patient ID'])):
        p = clinical['Patient ID'][i]
        if p in patient_to_pam50:
            clinical.loc[i, 'PAM50'] = patient_to_pam50[p]
    print(clinical.head())
    clinical.to_csv(c.CLINICAL_FILEPATH, index=False)


def format_numbers(nums):
    formatted_nums = []
    for n in nums:
        if n < 0.01:
            formatted_nums.append('{:.0e}'.format(n))
        else:
            formatted_nums.append('{:.2f}'.format(n))
    return formatted_nums
