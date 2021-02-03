from conf import *
from utils import *


def get_cohort_data_from_pan_cancer_data(pan_cancer_settings, patient_ids, out_suffix_name):
    out_file_name = pan_cancer_settings.out_subdata_format.format(out_suffix_name)
    if os.path.exists(out_file_name):
        return
    pan_cancer_df = pd.read_csv(pan_cancer_settings.filepath, sep=pan_cancer_settings.sep, nrows=1)
    pan_cancer_cols = list(pan_cancer_df.columns[1:])
    pan_cancer_cols_specific_patients = [pan_cancer_settings.col_name]
    for p in pan_cancer_cols:
        if 'TCGA' not in p:
            continue
        patient_id = p[:12]
        if patient_id in patient_ids and '-01A-' in p:  # only unhealthy samples (healthy will have >=10 instead of 01)
            pan_cancer_cols_specific_patients.append(p)

    # read all rows but only relevant patient id columns
    pan_cancer_specific_patients_df = pd.read_csv(pan_cancer_settings.filepath,
                                      sep=pan_cancer_settings.sep,
                                      usecols=pan_cancer_cols_specific_patients)
    print(pan_cancer_specific_patients_df.head())
    print("Saving as {}".format(out_file_name))
    pan_cancer_specific_patients_df.to_csv(out_file_name, index=False)


def sort_trait_by_highest_var_across_patients(pan_cancer_settings, filepath):
    trait_to_var = []
    df = pd.read_csv(filepath)
    for i in range(len(df[pan_cancer_settings.col_name])):
        row = df.iloc[i, :]
        row_trait = row[0]
        row_var = np.var(row[1:])
        trait_to_var.append((row_trait, row_var))
    trait_to_var = sorted(trait_to_var, key=lambda x: x[1], reverse=True)
    return trait_to_var


def add_expression_top_percentile_bottom_percentile(c, expression_file, gene_id, col_suffix='', bottom_percentile=None, top_percentile=None):
    def get_percentiles(values, percentiles):
        percentile_values = {}
        for p in percentiles:
            percentile_value = np.percentile(values, p)
            percentile_values[p] = percentile_value
        return percentile_values
    pan_cancer_gene_exp_subset = pd.read_csv(expression_file)
    gene_id_col_name = pan_cancer_gene_exp_subset.columns[0]
    print("Using column name {}".format(gene_id_col_name))
    gene_df = pan_cancer_gene_exp_subset[pan_cancer_gene_exp_subset[gene_id_col_name] == gene_id]
    gene_values = gene_df.values[0][1:]
    if bottom_percentile:
        max_val_negative_class = np.percentile(gene_values, bottom_percentile)
    if top_percentile:
        min_val_positive_class = np.percentile(gene_values, top_percentile)
    else:
        min_val_positive_class = max_val_negative_class + 1
    clinical = pd.read_csv(c.CLINICAL_FILEPATH)
    gene_label_col = gene_id + col_suffix
    clinical[gene_label_col] = np.nan # new column
    # check one value per subject
    subjects = [p[:12] for p in pan_cancer_gene_exp_subset.columns[1:]]
    assert len(set(list(subjects))) == len(pan_cancer_gene_exp_subset.columns[1:]), "More than one sample per subject!"
    patient_id_short_to_long = {p[:12] : p for p in pan_cancer_gene_exp_subset.columns[1:]}
    if 'value' in gene_label_col:
        percentiles = get_percentiles(gene_values, list(range(10, 101, 10)))
    for row_ix in range(len(clinical['Patient ID'])):
        row = clinical.iloc[row_ix]
        patient_id = row['Patient ID']
        if not patient_id in patient_id_short_to_long.keys():
            continue
        value = float(gene_df[patient_id_short_to_long[patient_id]])
        if 'value' in gene_label_col:
            for perc in list(range(10, 101, 10)):
                if value <= percentiles[perc]:
                    clinical.loc[row_ix, gene_label_col] = perc / 100
                    break
        else:
            if value >= min_val_positive_class:
                clinical.loc[row_ix, gene_label_col] = 'hi'
            if value <= max_val_negative_class:
                clinical.loc[row_ix, gene_label_col] = 'lo'
    print(clinical[gene_label_col].head())
    clinical.to_csv(c.CLINICAL_FILEPATH, index=False)


def get_gene_subset(pancan_settings, genes, genes_subset=None):
    if genes_subset is None:
        print(genes)
        return
    subset = []
    for g in genes:
        if pancan_settings.type == 'genes':
            if g[0].split('|')[0][:11] in genes_subset:
                subset.append(g)
        elif pancan_settings.type == 'mirs':
            mir_number = g[0].split('-')[2]
            for gs in genes_subset:
                if mir_number[:len(gs)] == gs:
                    subset.append(g)
    return subset


class PancanSettings:
    def __init__(self, type):
        self.type = type
        if type == 'genes':
            self.filepath = '../data/gene_exp_pan_cancer_normalized/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'
            self.col_name = 'gene_id'
            self.out_subdata_format = '{}.{}.csv'.format(self.filepath.split('.tsv')[0], '{}')
            self.sep = '\t'
        elif 'mirs' in type:
            self.filepath = '../data/miRNA/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv'
            self.col_name = 'Genes'
            self.out_subdata_format = '{}.{}.csv'.format(self.filepath.split('.csv')[0], '{}')
            self.sep = ','
        else:
            raise Exception("Undefined type: {}. Supported types: {}, {}.".format(type, 'mirs', 'genes'))


if __name__ == '__main__':

    c = Conf_BRCA()
    # c = Conf_LUAD()

    clinical_data = pd.read_csv(c.CLINICAL_FILEPATH)
    patients = list(clinical_data['Patient ID'])

    # MIRs finding gene candidates:
    # Pancan_Settings = PancanSettings('genes')
    Pancan_Settings = PancanSettings('mirs')
    get_cohort_data_from_pan_cancer_data(Pancan_Settings, patients, c.PANCAN_NAME_SUFFIX)
    cohort_data_expression_filepath = Pancan_Settings.out_subdata_format.format(c.PANCAN_NAME_SUFFIX)
    # EXPLORE:
    # genes = sort_trait_by_highest_var_across_patients(Pancan_Settings, cohort_data_expression_filepath)

    # MIRs adding chosen gene labels:
    if c.TCGA_COHORT_NAME == 'brca':
        chosen_genes = ['hsa-miR-17-5p', 'hsa-miR-29a-3p']
    else:
        chosen_genes = ['hsa-miR-17-5p', 'hsa-miR-21-5p']

    for gene in chosen_genes:
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene, bottom_percentile=20,
                                                        top_percentile=80)
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene,
                                                        col_suffix='_50_pctl', bottom_percentile=50)
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene,
                                                        col_suffix='_50_pctl_value', bottom_percentile=50)

    # Genes - finding gene candidates:
    Pancan_Settings = PancanSettings('genes')
    get_cohort_data_from_pan_cancer_data(Pancan_Settings, patients, c.PANCAN_NAME_SUFFIX)
    cohort_data_expression_filepath = Pancan_Settings.out_subdata_format.format(c.PANCAN_NAME_SUFFIX)
    # EXPLORE:
    # genes = sort_trait_by_highest_var_across_patients(Pancan_Settings, cohort_data_expression_filepath)

    # Genes adding chosen gene labels:
    if c.TCGA_COHORT_NAME == 'brca':
        chosen_genes = ['MKI67|4288', 'ERBB2|2064', 'CD24|100133941', 'ESR1|2099',
                        'EGFR|1956', 'FOXA1|3169', 'MYC|4609', 'FOXC1|2296']

    else:
        chosen_genes = ['EGFR|1956', 'KRAS|3845', 'CD274|29126']

    for gene in chosen_genes:
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene, bottom_percentile=20,
                                                        top_percentile=80)
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene,
                                                        col_suffix='_50_pctl', bottom_percentile=50)
        add_expression_top_percentile_bottom_percentile(c, cohort_data_expression_filepath, gene,
                                                        col_suffix='_50_pctl_value', bottom_percentile=50)