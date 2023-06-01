from models.VisualEncoder import VisualEncoder
from models.S3D_backbone import S3D_backbone
import torch
import time, os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
#import tensorflow as tf

from configs.VisualEncoderConfig import cfg
from train.trainer import validate, get_train_modules
from train_datasets.PHOENIXDataset import PhoenixDataset, collator, DataAugmentations
from train_datasets.preprocess_PHOENIX import getVocab, preprocess_df
from torchmetrics.functional import word_error_rate


def tokens_to_sent(vocab, tokens):
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)

def validate(model, dataloader, criterion, decoder, CFG, decode_func=None):

    ### setup validation metrics ###
    losses = []
    model.eval()
    start = time.time()
    word_error_rates = []
    secondary_word_error_rates = []

    ### iterature through dataloader ###
    for i, (ipt, vid_lens, trg, trg_len) in enumerate(dataloader):

        with torch.no_grad():
            ### get model output and calculate loss ###
            out, _ = model(ipt.to(CFG.device), vid_lens)
            x = out.permute(1, 0, 2)  
            ipt_len = torch.full(size=(1,), fill_value = out.size(1), dtype=torch.int32)
            loss = criterion(torch.log(x), 
                              trg, 
                              input_lengths=ipt_len,
                              target_lengths=trg_len) / out.size(0)
            
            ### save loss and get preds ###
            try:
                losses.append(loss.detach().cpu().item())
                out_d = decoder(out.cpu())
                preds = [p[0].tokens for p in out_d]
                pred_sents = [tokens_to_sent(CFG.gloss_vocab, s) for s in preds]
                ref_sents = tokens_to_sent(CFG.gloss_vocab, trg[0][:trg_len[0]])
                word_error_rates.append(word_error_rate(pred_sents, ref_sents).item())
                #secondary_word_error_rates.append(wer_list([ref_sents], pred_sents))
            except IndexError:
                print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")

            ### print iteration progress ###
            end = time.time()
            if max(1, i) % (CFG.val_print_freq) == 0:
                print("\n" + ("-"*10) + f"Iteration: {i}/{len(dataloader)}" + ("-"*10))
                print(f"Avg loss: {np.mean(losses):.6f}")
                print(f"Avg WER: {np.mean(word_error_rates):.4f}")
                #print(f"Avg Sec. WER: {np.mean(secondary_word_error_rates):.4f}")
                print(f"Time: {(end - start)/60:.4f} min")
                print(f"Predictions: {pred_sents}")
                print(f"References: {ref_sents}")
    
    ### print epoch progross ###
    print("\n" + ("-"*10) + f"VALIDATION" + ("-"*10))
    print(f"Avg WER: {np.mean(word_error_rates)}")
    #print(f"Avg WER Sec.: {np.mean(secondary_word_error_rates):.4f}")
    print(f"Avg loss: {np.mean(losses):.6f}")

    return losses, word_error_rates


class DataPaths:
  def __init__(self):
    self.phoenix_videos = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    self.phoenix_labels = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'

CFG = cfg()
dp = DataPaths()

### initialize data ###
train_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.train.corpus.csv'), delimiter = '|')#[:5]
val_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.dev.corpus.csv'), delimiter = '|')#[:5]
test_df = pd.read_csv(os.path.join(dp.phoenix_labels, 'PHOENIX-2014-T.test.corpus.csv'), delimiter = '|')#[:5]
### initialize data ###
PhoenixTrain = PhoenixDataset(train_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='train')
PhoenixVal = PhoenixDataset(val_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='dev')
PhoenixTest = PhoenixDataset(test_df, dp.phoenix_videos, vocab_size=CFG.VOCAB_SIZE, split='test')
### get dataloaders ###
train_augmentations = DataAugmentations(split_type='train')
val_augmentations = DataAugmentations(split_type='val')

dataloader_train = DataLoader(
      PhoenixTrain, 
      collate_fn = lambda data: collator(data, train_augmentations), 
      batch_size=CFG.batch_size, 
      shuffle=True, num_workers=CFG.num_workers)

dataloader_val = DataLoader(
    PhoenixVal, 
    collate_fn = lambda data: collator(data, val_augmentations), 
    batch_size=1, 
    shuffle=False,
    num_workers=CFG.num_workers)

dataloader_test = DataLoader(
    PhoenixTest, 
    collate_fn = lambda data: collator(data, val_augmentations), 
    batch_size=1, 
    shuffle=False,
    num_workers=CFG.num_workers)

print("HII")
### initialize model ###
model = VisualEncoder(CFG).to(CFG.device)

optimizer, criterion, scheduler, \
    decoder, train_losses, train_word_error_rates, \
    val_losses, val_word_error_rates = get_train_modules(model, dataloader_train, CFG)
print("HII")
### validate the model ###
validate(model, dataloader_test, criterion, decoder, CFG)