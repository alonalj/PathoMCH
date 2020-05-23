import os
import subprocess
import glob

'''
Useful post-training script to transfer model weights and predictions from server to GCP bucket.
'''

bucket_weights_preds_subfolder = 'brca'

files = glob.glob('../out/*/auc/*')  # get paths with files under auc subfolder
files = sorted(files, reverse=True)  # in case training was restarted at some point

results_folder_name = files[0].split('/')[2]
n_files_in_auc = len(os.listdir('../out/{}/auc/'.format(results_folder_name)))
assert n_files_in_auc > 0, "AUC folder empty"
predict_folders = os.listdir('../out/predict/')
predict_folders = [f for f in predict_folders if 'cv' in f]
assert len(predict_folders) == 1, "Predict folder contains more than one folder"
run_name = predict_folders[0].split('_')[0]
weights_name_format = '../out/weights_{}_{}.zip'.format(run_name, results_folder_name)
preds_name_format = '../out/preds_{}_{}.zip'.format(run_name, results_folder_name)
subprocess.run(['zip', '-r', weights_name_format, '../out/{}'.format(results_folder_name)])
subprocess.run(['zip', '-r', preds_name_format, '../out/predict/'.format(results_folder_name)])
subprocess.run(["gsutil", "cp", weights_name_format, 'gs://patho_al/weights/{}/'.format(bucket_weights_preds_subfolder)])
subprocess.run(["gsutil", "cp", preds_name_format, 'gs://patho_al/results/{}/'.format(bucket_weights_preds_subfolder)])

