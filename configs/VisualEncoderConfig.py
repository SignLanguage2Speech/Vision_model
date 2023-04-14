import os
from train_datasets.preprocess_PHOENIX import getVocab
import torch

### Config for training on Phoenix
class cfg:
    def __init__(self) -> None:
        self.n_classes = 1085 + 1 # +1 for blank token
        self.VOCAB_SIZE = self.n_classes - 1
        # S3D backbone
        self.use_block = 4 # use everything except lass block
        self.freeze = False
        # Head network
        self.ff_size = 2048
        self.input_size = 832
        self.hidden_size = 512
        self.ff_kernel_size = 3
        self.residual_connection = True
        # training 
        self.betas = (0.9, 0.998)
        self.weight_decay = 1e-3
        self.lr = 1e-3
        self.batch_size = 6
        self.n_epochs = 80
        self.num_workers = 8 #8
        self.print_freq = 100
        self.weights_filename = '/work3/s204138/bach-models/PHOENIX_trained_no_temp_aug/S3D_PHOENIX-21_epochs-5.337249_loss_0.983955_WER'
        self.backbone_weights_filename = 'WLASL/epoch299.pth.tar'
        self.head_weights_filename = None
        # verbose for weightloading #
        self.verbose = False
        self.start_epoch = 0
        ### paths ###
        #self.save_path = '/work3/s200925/VisualEncoder/checkpoints_BS6'
        self.save_path = '/work3/s204138/bach-models/PHOENIX_trained_no_temp_aug' # Michael save destination
        self.default_checkpoint = os.path.join(self.save_path, '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc')
        self.checkpoint_path = None #'/work3/s204138/bach-models/PHOENIX_trained_models/S3D_PHOENIX-19_epochs-5.361074_loss_0.999759_WER' # None  # if None train from scratch
        self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual')
        ### for data augmentation ###
        self.crop_size = 224
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
