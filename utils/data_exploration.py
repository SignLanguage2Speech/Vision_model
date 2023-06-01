import os
import numpy as np
import pandas as pd
import pdb
annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
test = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')
val = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
attributeNames = list(train.columns)

print(attributeNames)
print(train.head(5))
print(os.listdir(features_path))

chars = '?.,!-_+'

annotations = list(train['orth'])# + list(test['orth']) + list(val['orth'])
annotations = list(sorted(set([word for sent in annotations for word in sent.replace(chars,'').split(' ')])))
print(f"Train vocab size: {len(annotations)}")
pdb.set_trace()
#### Data preprocessing ####
full_df = pd.concat([train, test, val])


#print(str(full_df.loc[0]['orth']))

max = 0
for i in range(len(full_df)):
  max_new = len(str(full_df.iloc[i]['orth']).split(' '))
  if max_new > max:
    max = max_new
    #print("INDEXX", i)

print(f"Max gloss len: {max}")

max_train_len = 0
train_vid_names = os.listdir(os.path.join(features_path, 'train'))
test_vid_names = os.listdir(os.path.join(features_path, 'test'))
val_vid_names = os.listdir(os.path.join(features_path, 'dev'))

Lengths = []
for item in train_vid_names:
  Lengths.append(len(os.listdir(os.path.join(features_path, 'train', item))))

for item in test_vid_names:
  Lengths.append(len(os.listdir(os.path.join(features_path, 'test', item))))

for item in val_vid_names:
  Lengths.append(len(os.listdir(os.path.join(features_path, 'dev', item))))

print(f"mean: {np.mean(Lengths)}")
print(f"median: {np.median(Lengths)}")
print(f"std: {np.std(Lengths)}")


#print(train.iloc[2129]['name'])
#print(len(os.listdir(os.path.join(features_path, 'train/09August_2011_Tuesday_heute-2641'))))





    
