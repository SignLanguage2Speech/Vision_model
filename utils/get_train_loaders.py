

def getTrainLoaders(train_df, collator, dp, CFG):
    dataframes = preprocess_df(train_df, split='train', save=False)
    dataLoaders = []
    batch_sizes = [8, 8, 8, 8, 8, 8, 8,
                    8, 8, 8, 6, 4, 4, 2]
    
    for i, df in enumerate(dataframes):
        dataLoaders.append(DataLoader(PhoenixDataset(df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train'),
                                    batch_size=batch_sizes[i],
                                    shuffle=True, 
                                    num_workers=CFG.num_workers,
                                    collate_fn=collator))
    return dataLoaders