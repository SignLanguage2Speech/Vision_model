import os
import pandas as pd

import pdb

def preprocess_df(df, split, save=False, save_name = "PHOENIX_train_preprocessed.csv"):
    annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
    features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    gloss_vocab, translation_vocab = getVocab(annotations_path)

    # add translation and gloss labels
    df = getLabels(df, translation_vocab, gloss_vocab)

    # split dataframes
    if split == 'train':
       df = addLengths(df, features_path, cutoff=330)
       dataframes = groupByBin(df)
       return dataframes
       
    if save:
        df.to_csv(os.path.join(annotations_path, save_name))
    
    return df

def getLabels(df, t_vocab, g_vocab):

    all_translations = []
    all_glosses = []
    for i in range(len(df)):
        T = df.iloc[i]['translation'].split(' ')
        G = df.iloc[i]['orth'].split(' ')
        T_labels = []
        G_labels = []
        for word in T:
            try:
                T_labels.append(t_vocab[word])
            except KeyError:
                T_labels.append(1086) # TODO Figure out how to handle OOV in validation & test
        
        for gloss in G:
            try:
                G_labels.append(g_vocab[gloss])
            except KeyError:
                G_labels.append(1086) # TODO Figure out how to handle OOV in validation & test
                
        all_translations.append(T_labels)
        all_glosses.append(G_labels)
    
    df['translation_labels'] = all_translations
    df['gloss_labels'] = all_glosses

    return df

def getVocab(path):
    train = pd.read_csv(os.path.join(path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    chars = '?.,!-_+'

    # # get vocabulary for translations and glosses for the train dataset
    # glosses = list(train['orth']) #+ list(test['orth']) + list(val['orth'])
    # glosses = list(sorted(set([word for sent in glosses for word in sent.replace(chars,'').split(' ')])))
    # #print(f"Gloss vocab size: {len(glosses)}")

    # translations = list(train['translation']) #+ list(test['translation']) + list(val['translation'])
    # translations = list(sorted(set([word for sent in translations for word in sent.replace(chars,'').split(' ')])))
    
    # gloss_vocab = {word: glosses.index(word)+1 for word in glosses}
    # translation_vocab = {word: translations.index(word)+1 for word in translations}
    # return gloss_vocab, translation_vocab

    # get vocabulary for translations and glosses for the train dataset
    glosses = list(train['orth']) #+ list(test['orth']) + list(val['orth'])
    glosses = list(sorted(set([word for sent in glosses for word in sent.replace(chars,'').split(' ')])))
    #print(f"Gloss vocab size: {len(glosses)}")

    translations = list(train['translation']) #+ list(test['translation']) + list(val['translation'])
    translations = list(sorted(set([word for sent in translations for word in sent.replace(chars,'').split(' ')])))
    
    gloss_vocab = {word: glosses.index(word)+1 for word in glosses}
    translation_vocab = {word: translations.index(word)+1 for word in translations}
    return gloss_vocab, translation_vocab

def addLengths(df, features_path, cutoff=330):
  video_names = list(df['name'])
  lengths = []
  for name in video_names:
    length = len(os.listdir(os.path.join(features_path, 'train', name)))
    lengths.append(length)
  
  df['video_length'] = lengths
  df = df[df['video_length'] < cutoff]
  return df.reset_index()

def groupByBin(df):
  ### calculate bins
  min_val = min(df['video_length'])
  max_val = max(df['video_length'])
  bins = [min_val]
  val = min_val

  while val < max_val:
    val = int(val*1.25)
    bins.append(val)
# [16, 1.2*16, 1.2*1.2*16 ...]

  dataframes = []
  for i in range(len(bins)-1):
    df_new = df[(df['video_length'] >= bins[i]) & (df['video_length'] < bins[i+1])]
    dataframes.append(df_new)
  
  # pdb.set_trace()

  return dataframes
"""
########## TEST ##########
annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
features_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'

train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
val = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')
test = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')

val = preprocess_df(val)

print(val.head(5))

"""