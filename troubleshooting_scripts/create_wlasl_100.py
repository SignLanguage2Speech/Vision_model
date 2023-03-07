import os
import pdb
import pandas as pd

wlasl_wd = '/work3/s204503/bach-data/WLASL'
csv_file = 'WLASL_labels.csv'

df = {}
df = pd.read_csv(os.path.join(wlasl_wd, csv_file), delimiter=',')

video_files = os.listdir(os.path.join(wlasl_wd, 'WLASL100'))
video_ids = {int(vf[:-len('.mp4')]) for vf in video_files}

df = df.loc[df['video_id'].isin(video_ids)]

df.to_csv(os.path.join(wlasl_wd, 'WLASL100_labels.csv'))