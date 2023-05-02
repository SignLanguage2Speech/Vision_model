import torch

def save_checkpoint(path, model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS, config=None):
  torch.save({'epoch' : epoch,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'scheduler_state_dict' : scheduler.state_dict(),
              'train_losses' : train_losses,
              'val_losses' : val_losses,
              'train_WERS' : train_WERS,
              'val_WERS' : val_WERS,
              'config' : config
              }, path)