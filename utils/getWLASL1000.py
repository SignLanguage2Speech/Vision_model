import pandas as pd

df = pd.read_csv('data/WLASL/WLASL_labels.csv')
"""
Get dummy dataset for testing... contains 100 classes and 1000 samples
"""

def getWLASL1000(df):

    vocab = sorted(list(set(df['gloss'])))
    count_dict = {word : len(df.loc[df['gloss'] == word]) for word in vocab}
    count_dict = dict(sorted(count_dict.items(), key=lambda x:x[1]))
    WLASL1000 = {'gloss' : [word for word, count in count_dict.items() if count == 10][:100]}

    df = df.drop(columns=['label'])

    WLASL1000_df = df[df['gloss'].isin(WLASL1000['gloss'])].copy()

    # create labels
    words = list(sorted(set(WLASL1000_df['gloss'])))
    labels = []
    for i in range(len(WLASL1000_df)):
        labels.append(words.index(WLASL1000_df.iloc[i]['gloss']))

    WLASL1000_df['label'] = labels

    WLASL1000_df.to_csv('data/WLASL/WLASL1000_labels.csv')
    return WLASL1000_df


df = getWLASL1000(df)
"""
---- The code below shows that classes are equally represented in train and test ----

df = getWLASL1000(df)
dfTrain = df.loc[df['split'] == 'train']
dfVal = df.loc[df['split'] == 'val']

vocab = set(dfTrain['gloss'])
print(f"vocab size for train {len(vocab)}")


count_train = {word : len(dfTrain.loc[dfTrain['gloss'] == word]) for word in vocab}
print(count_train)

count_val = {word : len(dfVal.loc[dfVal['gloss'] == word]) for word in vocab}
print(count_val)

#print(len(df.loc[df['split'] == 'train']))
#print(len(df.loc[df['split'] == 'val']))
#print(len(df.loc[df['split'] == 'test']))
"""

