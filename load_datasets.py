import os
import itertools
import pdb
import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
from tensorflow_datasets.core.utils import gcs_utils
gcs_utils._is_gcs_disabled = True

SAVEPATH = '/work3/s204138/bach-data/'

dataset_names = ['rwth_phoenix2014_t', 'dicta_sign', 'chicago_fs_wild', 
                 'autsl', 'sign_bank', 'sign2_mint', 'swojs_glossario', 
                 'dgs_corpus', 'dgs_types','sign_suisse', 'ngt_corpus']
#dataset_names = ['chicago_fs_wild', 'autsl', 'sign_bank', 
#                 'sign2_mint', 'swojs_glossario', 'dgs_corpus', 'dgs_types', 
#                 'sign_suisse', 'ngt_corpus']

"""
dicta_sign:
    - word-level
    - splits: train (4154)
    - 
"""

import pandas as pd
def tfds2csv(tf_dataset : dict, save_path : str):
    attributes = list(tf_dataset['train'].element_spec.keys())
    for split in tf_dataset.keys(): # train, test, val
        values = {a : [] for a in attributes}
        for item in tf_dataset[split]:
            for a in attributes:
                values[a].append(item[a].numpy().decode('utf-8'))
        df = pd.DataFrame.from_dict(values)
        df.to_csv(save_path + f'/{split}.csv')

def getDatasets(ds_names):
    SAVEPATH = '/work3/s204138/bach-data/misc_datasets'
    config = SignDatasetConfig(name="only-annotations", version="1.0.0", include_video=False, include_pose=False)
    cnt = 0 
    for name in ds_names:
        if cnt > 0:
            break
        save_path = SAVEPATH + f'/{name}'
        os.chdir(SAVEPATH)
        print("CURRENT DIR: ", os.getcwd())
        tf_dataset = tfds.load(name=name, builder_kwargs={"config": config})
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        tfds2csv(tf_dataset, save_path)
        cnt+=1
    
if __name__ == '__main__':
    getDatasets(dataset_names)


