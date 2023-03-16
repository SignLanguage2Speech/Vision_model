import os
import pandas as pd

def preprocess_df(df, save=False, save_name = "PHOENIX_train_preprocessed.csv"):
    annotations_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
<<<<<<< Updated upstream
    train = pd.read_csv(os.path.join(annotations_path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    chars = '?.,!-_+'

    # get vocabulary for translations and glosses for the train dataset
    glosses = list(train['orth']) #+ list(test['orth']) + list(val['orth'])
    glosses = list(sorted(set([word for sent in glosses for word in sent.replace(chars,'').split(' ')])))
    print(f"Gloss vocab size: {len(glosses)}")

    translations = list(train['translation']) #+ list(test['translation']) + list(val['translation'])
    translations = list(sorted(set([word for sent in translations for word in sent.replace(chars,'').split(' ')])))
    
    gloss_vocab = {word: glosses.index(word)+1 for word in glosses}
    translation_vocab = {word: translations.index(word)+1 for word in translations}
=======
    gloss_vocab, translation_vocab = getVocab(annotations_path)
>>>>>>> Stashed changes

    # add translation and gloss labels
    df = getLabels(df, translation_vocab, gloss_vocab)

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
<<<<<<< Updated upstream
            except KeyError: # OOV
                T_labels.append(None)
=======
            except KeyError:
                T_labels.append(1086) # TODO Figure out how to handle OOV in validation & test
>>>>>>> Stashed changes
        
        for gloss in G:
            try:
                G_labels.append(g_vocab[gloss])
<<<<<<< Updated upstream
            except KeyError: # OOV
                G_labels.append(None)
=======
            except KeyError:
                G_labels.append(1086) # TODO Figure out how to handle OOV in validation & test
>>>>>>> Stashed changes
                
        all_translations.append(T_labels)
        all_glosses.append(G_labels)
    
    df['translation_labels'] = all_translations
    df['gloss_labels'] = all_glosses

    return df

def getVocab(path):
    train = pd.read_csv(os.path.join(path, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')
    chars = '?.,!-_+'

    # get vocabulary for translations and glosses for the train dataset
    glosses = list(train['orth']) #+ list(test['orth']) + list(val['orth'])
    glosses = list(sorted(set([word for sent in glosses for word in sent.replace(chars,'').split(' ')])))
    #print(f"Gloss vocab size: {len(glosses)}")

    translations = list(train['translation']) #+ list(test['translation']) + list(val['translation'])
    translations = list(sorted(set([word for sent in translations for word in sent.replace(chars,'').split(' ')])))
    
    gloss_vocab = {word: glosses.index(word)+1 for word in glosses}
    translation_vocab = {word: translations.index(word)+1 for word in translations}
    return gloss_vocab, translation_vocab

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
