from conf import *
from multiprocessing import Pool
from functools import partial
import shutil
from tfrecords_writer import *


log = Logger('../out/', 'preprocessor')


def create_image_path_to_labels_dict(c, create_image_path_to_labels_fn, n_examples_per_label=-1):
    '''
    creates and saves a dictionary mapping image paths to their sparse label (i.e. number representing class,
     not one-hot).
    :param
    create_image_path_to_labels_fn: a function that returns a dictionary mapping image paths to their sparse labels.
    n_examples_per_label: -1 to use all.
    :return:
    '''
    # should save a pkl file with the following dictionary: {img1_path_full : img1_label, img2_path_full: img2_label..}
    im_path_to_label_dict, labeled_slides = create_image_path_to_labels_fn(c, n_examples_per_label)
    save_obj(im_path_to_label_dict, c.IM_PATH_TO_LABEL_DICT_FORMAT.format(c.SLIDE_TYPE, '_'.join(c.LABELS)))
    save_obj(labeled_slides, c.IM_PATH_TO_LABEL_DICT_FORMAT.format(c.SLIDE_TYPE+'_labeled_slides_', '_'.join(c.LABELS)))
    return im_path_to_label_dict, labeled_slides


def create_image_path_to_labels_dict_slides(c, n_examples_per_label=-1):
    print("Creating image path to labels dict.")
    import pandas as pd

    def _get_mapping_label_to_pos_in_list():
        label_to_pos = {}
        for lix in range(len(c.LABELS)):
            label_to_pos[c.LABELS[lix]] = lix
        return label_to_pos
        
    def _get_pregenerated_labels(c):
        labels_string = c.IM_PATH_TO_LABEL_DICT_FORMAT.format(c.SLIDE_TYPE,'_'.join(c.LABELS))
        if os.path.exists('../res/{}.pkl'.format(labels_string)):
            print('Using EXISTING im_path_to_label_dict at: ../res/{}.pkl'.format(labels_string))
            return load_obj(labels_string)
        return {}

    def _combine_multiple_labels(c):
        label_dictionaries_to_include, skipped_slides = [], set([])
        im_path_to_label_dict = {}
        labeled_slides = set([])

        if c.CLINICAL_LABELS:
            clinical_patients_to_label_dict = \
                _get_clinical_labels(c)
            label_dictionaries_to_include.append('clinical')

        print("Including dictionaries: {}".format(' '.join(label_dictionaries_to_include)))
        for image_name in os.listdir(c.IMG_PATH):
            if not image_name.endswith(c.IMG_TYPE):
                continue
            slide_barcode = image_name[:c.N_CHAR_SLIDE_ID]
            im_label_vec = []
            sample_barcode = slide_barcode[:c.N_CHAR_SAMPLE_ID]
            patient_barcode = slide_barcode[:12]

            if 'clinical' in label_dictionaries_to_include:
                try:
                    sample_clinical_labels = clinical_patients_to_label_dict[patient_barcode]
                    im_label_vec.extend(sample_clinical_labels)
                    labeled_slides.add(sample_barcode)
                except:
                    # print("Clinical dictionary has no sample {} or its patient/sample id. "
                    #       "This may be expected if only LUAD/LUSC.".format(slide_barcode))
                    skipped_slides.add(slide_barcode)
                    continue

            im_path = c.IMG_PATH + image_name
            im_path_to_label_dict[im_path] = im_label_vec

        skipped_slides = list(skipped_slides)
        print("Skipped {} slides with missing labels. ".format(len(skipped_slides)))

        return im_path_to_label_dict, labeled_slides

    def _get_clinical_labels(c):
        print("Processing clinical labels.")
        patient_to_label_dict = {}
        d = pd.read_csv(c.CLINICAL_FILEPATH)
        cols = c.CLINICAL_LABEL_COLS.copy()
        if 'bcr_patient_barcode' in d.columns:
            patient_id_col = 'bcr_patient_barcode'
        elif 'Patient ID' in d.columns:
            patient_id_col = 'Patient ID'
        else:
            raise Exception("No Patient ID or bcr_patient_barcode columns in clinical data.")
        cols.append(patient_id_col)
        d = d[cols]
        d = d.dropna(axis=0)
        d = d.reset_index()
        print("Found {} rows for clinical cols {}".format(len(d), c.CLINICAL_LABEL_COLS))

        for column in c.CLINICAL_LABEL_COLS:
            labels = d[column]
            patients = d[patient_id_col]
            for p in patients:
                patient_to_label_dict[p] = []
            for pix in range(len(patients)):
                patient_to_label_dict[patients[pix]].append(labels[pix])
        patient_to_label_dict = _categories_to_single_bit_labels(c.CLINICAL_LABELS, patient_to_label_dict)
        save_obj(patient_to_label_dict, "patient_to_label_dict_clinical_debug")
        return patient_to_label_dict

    def _categories_to_single_bit_labels(categories, patient_to_label_dict):
        category_to_ix = {categories[c_ix]: c_ix for c_ix in range(len(categories))}
        for p in patient_to_label_dict.keys():
            patient_category = patient_to_label_dict[p][0]
            patient_to_label_dict[p] = [category_to_ix[patient_category]]
        return patient_to_label_dict

    im_path_to_label_dict = _get_pregenerated_labels(c)
    if len(im_path_to_label_dict.keys()) > 1 and c.USE_SAVED_LABELS_IF_EXIST:
        print("Found saved label dictionary for: {}. Skipping creating from scratch.".format(c.LABELS))
        return im_path_to_label_dict # todo - fix with cache

    im_path_to_label_dict, labeled_slides = _combine_multiple_labels(c)

    save_obj(im_path_to_label_dict, c.IM_PATH_TO_LABEL_DICT_FORMAT.format(c.SLIDE_TYPE, '_'.join(c.LABELS)))
    return im_path_to_label_dict, labeled_slides


def create_image_path_to_labels_dict_dummy_data(c, n_examples_per_label=-1):
    im_path_to_label_dict = {}
    for im_path in glob.glob(c.IMG_PATH + '*'):
        im_label = [-1]
        im_path_to_label_dict[im_path] = im_label

    return im_path_to_label_dict, {}


def create_train_val_test_patient_ids(patient_ids, train_pct, val_pct):
    '''
    all pct values should sum to 1 (if a test set is desired, set train_pct + val_pct < 1).
    :param patient_ids:
    :param train_pct:
    :param val_pct:
    :return:
    '''
    log.print_and_log("Running create_train_val_test_patient_ids")
    if os.path.exists('../res/train_patient_ids_round_0.pkl'):
        log.print_and_log("Already split patients into train, val, test. Using existing split.")
    else:
        log.print_and_log("No pre-split into train, val, test found!!! Splitting from scratch!")
        random.shuffle(patient_ids)
        n_patients = len(patient_ids)
        test_patient_ids = patient_ids[:int(n_patients * (1 - train_pct - val_pct))]
        if len(test_patient_ids) == 0:
            log.print_and_log('No test patients!')
        save_obj(test_patient_ids, 'test_patient_ids')
        test_patient_ids = set(test_patient_ids)
        train_val_patient_ids = [p for p in patient_ids if p not in test_patient_ids]
        n_train_val_patient_ids = len(train_val_patient_ids)
        log.print_and_log('Total number of patients for training and validation is {}'.format(n_train_val_patient_ids))
        print(train_val_patient_ids)
        for resample_round in range(c.N_ROUNDS):
            random.shuffle(train_val_patient_ids)
            train_patient_ids = train_val_patient_ids[:int(n_train_val_patient_ids * train_pct)]
            val_patient_ids = train_val_patient_ids[int(n_train_val_patient_ids * train_pct):]
            assert len([p for p in train_patient_ids if (p in val_patient_ids or p in test_patient_ids)]) == 0
            assert len([p for p in val_patient_ids if p in test_patient_ids]) == 0
            save_obj(val_patient_ids, 'val_patient_ids_round_{}'.format(resample_round))
            save_obj(train_patient_ids, 'train_patient_ids_round_{}'.format(resample_round))


def create_train_val_test_dictionary(c, im_path_to_label_dict, labeled_slides, create_train_val_test_patient_ids_fn, remove_damaged_slides):
    '''
    creates a general train val test split of slides.
    :param c: 
    :param train_pct: 
    :param val_pct: 
    :return: 
    '''
    print("Creating train val test dict")

    def _get_unique_patient_ids_with_images():
        slide_names = glob.glob(c.IMG_PATH + '*')
        patient_ids = list(set([get_minimal_slide_identifier(s)[:c.N_CHAR_PATIENT_ID] for s in slide_names]))
        log.print_and_log("n unique patient ids = {}".format(len(patient_ids)))
        return patient_ids

    def _get_unique_sample_ids_with_images():
        slide_names = glob.glob(c.IMG_PATH + '*')
        sample_ids = list(set([get_minimal_slide_identifier(s)[:c.N_CHAR_SAMPLE_ID] for s in slide_names]))
        log.print_and_log("n unique available sample ids = {}".format(len(sample_ids)))
        return sample_ids

    def _get_train_val_test_sample_ids(sample_ids):
        '''
        splits patient ids into train val and test sets so samples from any one patient are all in same subset
        :param sample_ids_pos:
        :param sample_ids_neg:
        :return:
        '''
        resample_round_to_train_val_test_split = {}

        log.print_and_log("Loading train/val/test patient ids.")

        test_patient_ids = load_obj('test_patient_ids')
        test_sample_ids = [s for s in sample_ids if s[:c.N_CHAR_PATIENT_ID] in test_patient_ids]

        # train val per resampling round
        for resample_round in range(c.N_ROUNDS):
            train_patient_ids = load_obj('train_patient_ids_round_{}'.format(resample_round))
            val_patient_ids = load_obj('val_patient_ids_round_{}'.format(resample_round))
            train_sample_ids = [s for s in sample_ids if s[:c.N_CHAR_PATIENT_ID] in train_patient_ids]
            val_sample_ids = [s for s in sample_ids if s[:c.N_CHAR_PATIENT_ID] in val_patient_ids]
            save_obj(train_sample_ids, 'train_sample_ids_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))
            save_obj(val_sample_ids, 'val_sample_ids_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))

            resample_round_to_train_val_test_split[resample_round] = [train_sample_ids, val_sample_ids, test_sample_ids]

        return resample_round_to_train_val_test_split

    if os.path.exists('../res/test_img_paths_{}.pkl'.format(c.SLIDE_TYPE)):
        log.print_and_log("Already split image paths into general train, val, test sets.")
        return

    patient_ids = _get_unique_patient_ids_with_images()
    patient_ids_set = set(patient_ids)
    sample_ids = _get_unique_sample_ids_with_images()
    available_labeled_samples = [s for s in sample_ids
                                 if s[:c.N_CHAR_PATIENT_ID] in patient_ids_set and
                                 s[:c.N_CHAR_SAMPLE_ID] in labeled_slides]
    log.print_and_log("available labeled samples: {}".format(available_labeled_samples))

    create_train_val_test_patient_ids_fn(patient_ids, c.TRAIN_PCT, c.VAL_PCT)
    round_to_train_val_test_dict = _get_train_val_test_sample_ids(available_labeled_samples)

    for resample_round in range(c.N_ROUNDS):
        train_sample_ids, val_sample_ids, test_sample_ids = round_to_train_val_test_dict[resample_round]

        # verifying none of train patient ids are in val or test
        mutual_patients_train_test = [s for s in train_sample_ids if s in test_sample_ids]
        mutual_samples_train_val = [s for s in train_sample_ids if s in val_sample_ids]
        assert len(mutual_patients_train_test) == 0 and len(mutual_samples_train_val) == 0

        log.print_and_log("Number of samples in each sub data: (numbers may\n "
                          "vary if some patients have multiple samples)")
        log.print_and_log("n samples val = {}".format(len(val_sample_ids)))
        log.print_and_log("n samples train = {}".format(len(train_sample_ids)))
        log.print_and_log("n samples test = {}".format(len(test_sample_ids)))

        img_association_sub_data = {}
        for sample_id in train_sample_ids:
            img_association_sub_data[sample_id] = 'train'
        for sample_id in val_sample_ids:
            img_association_sub_data[sample_id] = 'val'
        for sample_id in test_sample_ids:
            img_association_sub_data[sample_id] = 'test'

        img_paths = glob.glob(c.IMG_PATH+'*')
        train_img_paths, val_img_paths, test_img_paths = [], [], []
        for im_path in img_paths:
            slide_id = get_minimal_slide_identifier(im_path)
            if slide_id[:c.N_CHAR_SAMPLE_ID] in img_association_sub_data.keys():
                if img_association_sub_data[slide_id[:c.N_CHAR_SAMPLE_ID]] == 'train':
                        train_img_paths.append(im_path)
                elif img_association_sub_data[slide_id[:c.N_CHAR_SAMPLE_ID]] == 'val':
                    val_img_paths.append(im_path)
                else:
                    test_img_paths.append(im_path)

        save_obj(train_img_paths, 'train_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))
        save_obj(val_img_paths, 'val_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))
    save_obj(test_img_paths, 'test_img_paths_{}'.format(c.SLIDE_TYPE))


def get_minimal_slide_identifier(slide_string):
    '''

    :param slide_string: any string that contains the slide name
    :return:
    '''
    if '_' in slide_string:
        return slide_string.split('/')[-1].split('_')[0]
    else:
        return slide_string.split('/')[-1].split('.')[0]


def get_svs_paths():
    svs_paths = []
    for dirpath, subdirs, files in os.walk(c.SVS_SLIDES_PATH):
        for f in files:
            if f.endswith('.svs'):
                slide_id = get_minimal_slide_identifier(f)
                slidepath = dirpath + '/' + f
                svs_paths.append(slidepath)
    return svs_paths


def tile_slides_multiprocess(c):
    from deep_zoom import deep_zoom_tile
    '''
    Tile slides with white removal.
    '''
    slide_paths = get_svs_paths()

    if not os.path.exists(c.IMG_PATH):
        os.makedirs(c.IMG_PATH)
    with Pool(c.NUM_CPU) as p:
        p.map(deep_zoom_tile, slide_paths)


def create_sample_tfrecord_multiprocess(sample_id_to_img_paths, sample_id):
    ''' can accept (img_path, label) tuples'''
    sample_items = []
    for img_path in sample_id_to_img_paths[sample_id].keys():
        sample_items.append((img_path, sample_id_to_img_paths[sample_id][img_path]))
    tfrecords_name = c.ALL_SAMPLES_TFRECORDS_FOLDER + "{}.tfrecords".format(sample_id)
    tfrec_writer.save_tfrecords(sample_items, tfrecords_name)


def create_im_path_to_sample_id_multiprocess(c, img_path_to_label_dict):
    from multiprocessing import Process, Manager

    def create_im_path_to_sample_id_dict(_im_path_to_sample_id, img_paths):
        for im_path in img_paths:  # path to tile
            sample_id = get_minimal_slide_identifier(im_path)
            _im_path_to_sample_id[im_path] = sample_id
    log.print_and_log("running create_im_path_to_sample_id_multiprocess")
    manager = Manager()
    im_path_to_sample_id = manager.dict()
    processes = []
    img_paths = list(img_path_to_label_dict.keys())
    n_per_cpu = len(img_paths)//c.NUM_CPU
    if n_per_cpu == 0:
        n_per_cpu = len(img_paths)
    for i in range(0, c.NUM_CPU):
        if i == c.NUM_CPU -1:
            processes.append(
                Process(target=create_im_path_to_sample_id_dict, args=(im_path_to_sample_id, img_paths[i:])))
        else:
            processes.append(
                Process(target=create_im_path_to_sample_id_dict,
                        args=(im_path_to_sample_id, img_paths[i*n_per_cpu: (i+1)*n_per_cpu])))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    log.print_and_log("Done")
    return im_path_to_sample_id


def create_tfrecord_per_sample(c, img_path_to_label_dict, subdata_type):

    sample_id_to_img_paths = {}
    tfrecord_paths = []

    im_path_to_sample_id = create_im_path_to_sample_id_multiprocess(c, im_path_to_label_dict)
    log.print_and_log("Preparing for tfrecords - creating sample_id_to_img_paths dictionary")
    count = 0
    for im_path in img_path_to_label_dict.keys():  # path to tile

        sample_id = im_path_to_sample_id[im_path]
        try:
            sample_id_to_img_paths[sample_id][im_path] = img_path_to_label_dict[im_path]
        except:
            sample_id_to_img_paths[sample_id] = {}
            sample_id_to_img_paths[sample_id][im_path] = img_path_to_label_dict[im_path]
        if count % 1000 == 0:
            log.print_and_log("processed {} img paths".format(count))
        count += 1

    with Pool(c.NUM_CPU) as p:
        func = partial(create_sample_tfrecord_multiprocess, sample_id_to_img_paths)  # sample_id_to_img_label_tup
        p.map(func, sample_id_to_img_paths.keys())

    for sample_id in sample_id_to_img_paths.keys():
        tfrecords_name = c.ALL_SAMPLES_TFRECORDS_FOLDER + "{}.tfrecords".format(sample_id)
        tfrecord_paths.append(tfrecords_name)

    print(tfrecord_paths)
    save_obj(tfrecord_paths, 'per_sample_tfrecord_paths_{}'.format(subdata_type), c.ALL_SAMPLES_TFRECORDS_FOLDER)
    return tfrecord_paths


def get_subdata_im_path_to_label_dict(img_path_to_label_dict, chosen_img_paths):
    sub_data_im_path_to_label_dict = {}
    count = 0
    for im_path in chosen_img_paths:  # path to tile
        try:
            sub_data_im_path_to_label_dict[im_path] = img_path_to_label_dict[im_path]
            count += 1
        except:
            # log.print_and_log("im_path {} not in label dictionary. This may be expected if using part of data.")
            continue
        if count % 100 == 0:
            print(count)
    return sub_data_im_path_to_label_dict


def create_example_tfrecords(c, resample_round):

    from tfrecords_reader import tfrecords
    tf.enable_eager_execution()
    files_train = glob.glob('../res/train/*round_{}*train*.tfrec'.format(resample_round))
    random.shuffle(files_train)
    files_val = glob.glob('../res/val/*round_{}*val*.tfrec'.format(resample_round))
    random.shuffle(files_val)
    sample_file_train, sample_file_val = files_train[0], files_val[0]
    for sample_data, sample_data_name in [(sample_file_train, 'train'), (sample_file_val, 'val')]:
        TFRec = tfrecords(c, with_name=True).get_batched_dataset([sample_data], 1)
        count = 0
        for img, label, name in TFRec:
            img, label, name = img.numpy()[0], label.numpy()[0], name.numpy()[0]
            if img.shape[-1] == 1:
                # grayscale
                img = img[:, :, 0]
            img = Image.fromarray((img*255).astype('uint8'))
            label = list(label)
            name = name.decode('utf-8')
            if len(label) == 1:
                img.save('../out/sample_image_{}_round_{}_label_{}_name_{}'.format(sample_data_name, resample_round, label, name))
            else:
                img.save('../out/sample_image_{}_round_{}_unet_name_{}'.format(sample_data_name, resample_round, name))
            count += 1
            if count == 10:
                break
        log.print_and_log("Saved examples from train tfrecords under out folder. ")


def del_train_val_test_img_copies():
    for p in ['../data/train/','../data/val/', '../data/test/']:
        try:
            shutil.rmtree(p)
        except:
            pass


def cleanup_previous_run(remove_patient_ids_master_split):
    to_remove = glob.glob('../res/im_path*.pkl')
    to_remove.extend(glob.glob('../res/*img_path*.pkl'))
    if remove_patient_ids_master_split:
        to_remove.extend(glob.glob('../res/*patient_id*.pkl'))
        log.print_and_log("Cleaning up ALL files from previous run, including train, val, test patient splits.")
    else:
        log.print_and_log("Cleaning up files from previous run, EXCEPT for train, val, test patient splits.")
    to_remove.extend(glob.glob('../res/*sample_id*.pkl'))
    to_remove.extend(glob.glob('../res/*tfrec*'))

    for file_path in to_remove:
        os.remove(file_path)

    del_train_val_test_img_copies()


if __name__ == '__main__':

    # first use gdc-client to obtain slides with hith manifest:  # TODO
    # ./gdc-client download -m gdc_manifest_20190507_125211.txt

    c = Conf_BRCA_TRAITS_miR_17_5p_extreme()
    # c = Conf_BRCA_DUMMY_LABEL()  # used to generate a tfrecord per sample for post-training predictions

    cleanup = True  # True if you're ready to move on to a new trait and don't want and tfrec / tfrecords etc. left.
    remove_patient_ids_master_split = False  # False will use ..patient_ids..pkl found under res to split.
    tile_slides = True  # turn to False if you no longer want it to tile slides (e.g. new trait, but same slides)

    create_img_path_to_labels_dict = True
    if c.NAME == 'Dummy':
        create_per_sample_tf_records = True
        create_tf_records_and_labels = False
    else:
        create_per_sample_tf_records = False
        create_tf_records_and_labels = True

    if cleanup:
        cleanup_previous_run(remove_patient_ids_master_split)

    if tile_slides:
        tile_slides_multiprocess(c)
    print(c.IMG_SIZE, c.IMG_PATH)
    im_path_to_labels_fn = create_image_path_to_labels_dict_slides
    split_train_val_test_fn = create_train_val_test_patient_ids

    if create_img_path_to_labels_dict:
        im_path_to_label_dict, labeled_slides = create_image_path_to_labels_dict(c, im_path_to_labels_fn, n_examples_per_label=-1)

    log.print_and_log("finished im_path_to_label")

    create_train_val_test_dictionary(c, im_path_to_label_dict, labeled_slides, split_train_val_test_fn, remove_damaged_slides=True)

    if create_tf_records_and_labels:
        tfrec_writer = tfrecords_writer(c)
        pool = multiprocessing.Pool(c.NUM_CPU)

        for resample_round in range(c.N_ROUNDS):
            print("round", resample_round)

            train_img_paths = load_obj('train_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))
            val_img_paths = load_obj('val_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, resample_round))
            test_img_paths = load_obj('test_img_paths_{}'.format(c.SLIDE_TYPE))

            for (sub_data_filepaths, sub_data_name) in \
                    [(train_img_paths, "train"), (val_img_paths, "val")]:

                random.shuffle(sub_data_filepaths)

                log.print_and_log(sub_data_name)
                if "train" in sub_data_name:
                    SHARDS = 100
                else:
                    SHARDS = 20

                n_images = len(sub_data_filepaths)
                log.print_and_log("{} has {} images".format(sub_data_name, n_images))
                shard_size = math.ceil(1.0 * n_images / SHARDS)

                shard_id_to_shard_filenames = {}
                for i in range(SHARDS):
                    shard_id_to_shard_filenames[i] = sub_data_filepaths[i * shard_size:(i + 1) * shard_size]

                pool.map(functools.partial(tfrec_writer.create_single_tfrecord_shard,
                                           shard_id_to_shard_filenames=shard_id_to_shard_filenames,
                                           im_path_to_label_dict=im_path_to_label_dict,
                                           sub_data_name=sub_data_name,
                                           resample_round=resample_round
                                           ),
                         shard_id_to_shard_filenames.keys())

                log.print_and_log("Finished saving images as tfrecords {}".format(sub_data_name))

            create_example_tfrecords(c, resample_round)
            log.print_and_log("DONE! Zoom {} img size {} round {}.".format(c.ZOOM_LEVEL, c.IMG_SIZE, resample_round))

    if create_per_sample_tf_records:
        tfrec_writer = tfrecords_writer(c)
        # load paths from any round (per sample is round agnostic)
        train_img_paths = load_obj('train_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, 0))
        val_img_paths = load_obj('val_img_paths_{}_round_{}'.format(c.SLIDE_TYPE, 0))
        test_img_paths = load_obj('test_img_paths_{}'.format(c.SLIDE_TYPE))
        for (sub_data_filepaths, sub_data_name) in \
                [(train_img_paths, "train"), (val_img_paths, "val"), (test_img_paths, "test")]:
            im_path_to_label_dict_subdata = get_subdata_im_path_to_label_dict(im_path_to_label_dict, sub_data_filepaths)
            tf_records_paths = create_tfrecord_per_sample(c, im_path_to_label_dict_subdata, sub_data_name)
            log.print_and_log("Finished per sample tfrecords {}".format(sub_data_name))







