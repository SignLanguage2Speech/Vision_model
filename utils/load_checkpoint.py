import torch

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_WERS = checkpoint['train_WERS']
    val_WERS = checkpoint['val_WERS']

    if "config" in checkpoint.keys():
        config = checkpoint['config']
        return model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS, config
    else:
        return model, optimizer, scheduler, epoch, train_losses, val_losses, train_WERS, val_WERS, None