import os
import numpy as np
import time
import random

import torch
import torch.optim as optim
from torchaudio.models.decoder import ctc_decoder
from torchmetrics.functional import word_error_rate

from utils.load_weigths import load_PHOENIX_weights
from utils.load_checkpoint import load_checkpoint
from utils.tokens_to_sent import tokens_to_sent
from utils.save_checkpoint import save_checkpoint
from utils.secondary_word_error_rate import wer_list

def get_train_modules(model, dataloader_train, CFG):

    ### get optimizer and scheduler ###
    optimizer = optim.AdamW(
        model.parameters(),
        lr = CFG.lr,
        betas = CFG.betas,
        weight_decay = CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = CFG.n_epochs)

    ### load checkpoint of model weights ###
    if CFG.checkpoint_path is not None:
        model, optimizer, scheduler, current_epoch, \
        train_losses, val_losses, train_word_error_rates, \
        val_word_error_rates = load_checkpoint(CFG.checkpoint_path, model, optimizer, scheduler)
        CFG.start_epoch = current_epoch
    else:
        train_losses = []
        train_word_error_rates = []
        val_losses = []
        val_word_error_rates = []

    ### get decoder and criterion ###
    CTC_decoder = ctc_decoder(
        lexicon=None,       
        lm_dict=None,       
        lm=None,            
        tokens= ['-'] + [str(i+1) for i in range(CFG.VOCAB_SIZE)] + ['|'], # vocab + blank and split
        nbest=1, # number of hypotheses to return
        beam_size = 100,       # n.o competing hypotheses at each step
        beam_size_token=25,  # top_n tokens to consider at each step
    )
    criterion = torch.nn.CTCLoss(
        blank=0, 
        zero_infinity=True, 
        reduction='mean'
    ).to(CFG.device)

    ### send model to device ###
    model.to(CFG.device)

    return optimizer, criterion, scheduler, CTC_decoder, train_losses, train_word_error_rates, val_losses, val_word_error_rates


def train(model, dataloader_train, dataloader_val, CFG):

    ### printing training ###
    print("-"*10 + "STARTING TRAINING OF BS" + str(CFG.batch_size) + "-"*10)
    
    ### initialize training modules ###
    optimizer, criterion, scheduler, \
    decoder, train_losses, train_word_error_rates, \
    val_losses, val_word_error_rates = get_train_modules(model, dataloader_train, CFG)

    ### iterate through specified epochs ###
    for epoch in range(CFG.start_epoch, CFG.n_epochs):

        ### initialize epoch variables ###
        losses = []
        model.train()
        start = time.time()
        word_error_rates = []

        ### iterate through each batch in dataloader ###
        for itt, (ipt, _, trg, trg_len) in enumerate(dataloader_train):
            refs = [t[:trg_len[i]].cpu() for i, t in enumerate(trg)]
            ref_sents = [tokens_to_sent(CFG.gloss_vocab, s) for s in refs]

            ### get output and calculate loss ###
            out = model(ipt.to(CFG.device))
            x = out.permute(1, 0, 2)
            trg = torch.concat(refs).to(CFG.device)
            trg_len = trg_len.to(torch.int32)
            ipt_len = torch.full(size=(out.size(0),), fill_value = out.size(1), dtype=torch.int32)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = criterion(torch.log(x), 
                                trg,
                                input_lengths=ipt_len,
                                target_lengths=trg_len)
            
            ### backprop and steps ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ### save loss and get preds ###
            try:
                losses.append(loss.detach().cpu().item())
                out_d = decoder(out.cpu())
                preds = [p[0].tokens for p in out_d]
                pred_sents = [tokens_to_sent(CFG.gloss_vocab, s) for s in preds]
                word_error_rates.append(word_error_rate(pred_sents, ref_sents).item())
            except IndexError:
                print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")
            
            ### print iter progross ###
            end = time.time()
            if max(1, itt) % (CFG.train_print_freq) == 0:
                print("\n" + ("-"*10) + f"Iteration: {itt}/{len(dataloader_train)}" + ("-"*10))
                print(f"Avg loss: {np.mean(losses):.6f}")
                print(f"Avg WER: {np.mean(word_error_rates):.4f}")
                print(f"Time: {(end - start)/60:.4f} min")
                print(f"Predictions: {pred_sents}")
                print(f"References: {ref_sents}")

        ### print epoch progross ###
        print("\n" + ("-"*10) + f"EPOCH {epoch}" + ("-"*10))
        print(f"Avg WER: {np.mean(word_error_rates)}")
        print(f"Avg loss: {np.mean(losses):.6f}")
        
        ### save epoch metrics ###
        train_losses.append(losses)
        train_word_error_rates.append(word_error_rates)

        ### run validation loop ###
        val_loss, val_word_error_rate = validate(model, dataloader_val, criterion, decoder, CFG)
        val_losses.append(val_loss)
        val_word_error_rates.append(val_word_error_rate)

        ### Save checkpoint ###
        fname = os.path.join(CFG.save_path, f'S3D_PHOENIX-{epoch+1}_epochs-{np.mean(val_loss):.6f}_loss_{np.mean(val_word_error_rate):5f}_WER')
        save_checkpoint(fname, model, optimizer, scheduler, epoch+1, train_losses, val_losses, train_word_error_rates, val_word_error_rates)

        ### stepping with scheduler ###
        scheduler.step()



def validate(model, dataloader, criterion, decoder, CFG, decode_func=None):

    ### setup validation metrics ###
    losses = []
    model.eval()
    start = time.time()
    word_error_rates = []
    secondary_word_error_rates = []

    ### iterature through dataloader ###
    for i, (ipt, _, trg, trg_len) in enumerate(dataloader):

        with torch.no_grad():
            ### get model output and calculate loss ###
            out = model(ipt.to(CFG.device))
            x = out.permute(1, 0, 2)  
            ipt_len = torch.full(size=(1,), fill_value = out.size(1), dtype=torch.int32)
            loss = criterion(torch.log(x), 
                              trg, 
                              input_lengths=ipt_len,
                              target_lengths=trg_len)
            
            ### save loss and get preds ###
            try:
                losses.append(loss.detach().cpu().item())
                out_d = decoder(out.cpu())
                preds = [p[0].tokens for p in out_d]
                pred_sents = [tokens_to_sent(CFG.gloss_vocab, s) for s in preds]
                ref_sents = tokens_to_sent(CFG.gloss_vocab, trg[0][:trg_len[0]])
                word_error_rates.append(word_error_rate(pred_sents, ref_sents).item())
                secondary_word_error_rates.append(wer_list(ref_sents,pred_sents))
            except IndexError:
                print(f"The output of the decoder:\n{out_d}\n caused an IndexError!")

            ### print iteration progress ###
            end = time.time()
            if max(1, i) % (CFG.val_print_freq) == 0:
                print("\n" + ("-"*10) + f"Iteration: {i}/{len(dataloader)}" + ("-"*10))
                print(f"Avg loss: {np.mean(losses):.6f}")
                print(f"Avg WER: {np.mean(word_error_rates):.4f}")
                print(f"Avg Sec. WER: {np.mean(secondary_word_error_rates):.4f}")
                print(f"Time: {(end - start)/60:.4f} min")
                print(f"Predictions: {pred_sents}")
                print(f"References: {ref_sents}")
    
    ### print epoch progross ###
    print("\n" + ("-"*10) + f"VALIDATION" + ("-"*10))
    print(f"Avg WER: {np.mean(word_error_rates)}")
    print(f"Avg WER Sec.: {np.mean(secondary_word_error_rates):.4f}")
    print(f"Avg loss: {np.mean(losses):.6f}")

    return losses, word_error_rates