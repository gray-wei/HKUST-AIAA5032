import os
import shutil
import pandas as pd

# folder path, the data is from https://www.kaggle.com/competitions/hkustgz-aiaa-5032-hw-1-spring-2025/overview
data_folder = 'mfcc'  
trainval_csv_path = 'labels/trainval.csv'  
test_label_path = 'labels/test_for_student.label'  
trainval_folder = 'data/trainval'  
test_folder = 'data/test'  

# create target folder
os.makedirs(trainval_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# read trainval.csv
trainval_df = pd.read_csv(trainval_csv_path)

# read test_for_student.label file
with open(test_label_path, 'r') as f:
    test_files = [line.strip() for line in f.readlines()]

# get all csv files
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.mfcc.csv')]

# dictionary for file name and corresponding label (trainval)
trainval_dict = dict(zip(trainval_df['Id'], trainval_df['Category']))

# process trainval data
for csv_file in csv_files:
    file_id = csv_file.split('.')[0]  # extract Id from file name, assume the format is HW00006645.mfcc.csv
    if file_id in trainval_dict:
        label = trainval_dict[file_id]
        new_name = f"{file_id}_{label}.mfcc.csv"
        shutil.copy(os.path.join(data_folder, csv_file), os.path.join(trainval_folder, new_name))

# process test data
for test_file in test_files:
    test_id = test_file.split('.')[0]  # extract Id from file name, assume the format is HW00006645.mp4
    corresponding_csv = f"{test_id}.mfcc.csv"
    if corresponding_csv in csv_files:
        shutil.copy(os.path.join(data_folder, corresponding_csv), os.path.join(test_folder, corresponding_csv))

print("File classification completed!")