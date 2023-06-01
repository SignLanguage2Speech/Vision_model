import os
from train_datasets.preprocess_PHOENIX import getVocab
import torch

"""
###### ORIGINAL AUTHORS S2G CFG ######

transform_cfg:
    img_size: 224
    aug_hflip: false
    color_jitter: true
    bottom_area: 0.7
    center_crop: true
    center_crop_size: 270
    randomcrop_threshold: 1
    aspect_ratio_min: 0.75
    aspect_ratio_max: 1.3
    temporal_augmentation:
      tmin: 0.5
      tmax: 1.5

optimization:
    optimizer: Adam
    learning_rate:
      default: 1.0e-3
    weight_decay: 0.001
    betas:
    - 0.9
    - 0.998
    scheduler: cosineannealing
    t_max: 40

model:
  RecognitionNetwork:
    GlossTokenizer:
      gloss2id_file: data/csl-daily/gloss2ids.pkl
    s3d:
      pretrained_ckpt: pretrained_models/s3ds_glosscls_ckpt
      use_block: 4
      freeze_block: 1
    visual_head:
      input_size: 832
      hidden_size: 512
      ff_size: 2048 
      pe: True
      ff_kernelsize:
        - 3
        - 3

####### USING THEIR EXACT CONFIG CURRENTLY, THIS INCLUDES: 
 - MaskedNorm in place of self.bn1 in HeadNetwork
 - Adam instead of AdamW
 - #with torch.backends.cudnn.flags(enabled=False): --> removed in trainer.py during loss computation for train()
 - CTC Loss: Blank=0, zero_inf = True, reduction='sum'
 - Without HHLIP augmentation
 - #torch.backends.cudnn.deterministic = True --> removed in PHOENIX_main line 35


####### EXCEPTIONS
 - Use_block = 0 (instead of 1)
 - Using dropout 0.15
 - Using their weights (???) --> No for now
 - With random rotation
 - 100 epochs instead of 40, T_max = n_epochs

 
 ######## ABOVE DID NOT WORK... CANNOT OVERFIT ON FEW SAMPLES --> BC OF MASKING
 Running again without masking and dropout 0.10, otherwise unchanged


 ####### Kinetics training Config!

  - Dropout = 0.10
  - Everything unfrozen
  - AdamW optimizer
  - torch.backends.cudnn.deterministic = True
  - 100 epochs, T_max = 100
  - Without Hflip
  - Starting with kinetics
  - residual_connection = True
"""

### Config for training on Phoenix
class cfg:
    def __init__(self) -> None:
        self.n_classes = 2391 + 1 # +1 for blank token # 1085 
        self.VOCAB_SIZE = self.n_classes - 1
        # S3D backbone
        self.use_block = 4 # use everything except lass block
        self.freeze_block = 0 # [0, ...5] 
        # Head network
        self.ff_size = 2048
        self.input_size = 832
        self.hidden_size = 512
        self.ff_kernel_size = 3
        self.residual_connection = True
        self.head_dropout = 0.10 # 0.10 in SOTA config
        # training
        self.betas = (0.9, 0.998)
        self.weight_decay = 1.0e-3
        self.lr = 1.0e-3
        self.batch_size = 1
        self.n_epochs = 150
        self.num_workers = 8
        self.train_print_freq = 10
        self.val_print_freq = 50
        # verbose for weightloading #
        self.verbose = True # Verbose weight loading
        self.start_epoch = 0
        ### paths ###
        # self.weights_filename = '/work3/s204138/bach-models/PHOENIX_trained_no_temp_aug/S3D_PHOENIX-21_epochs-5.337249_loss_0.983955_WER'
        #self.backbone_weights_filename = '/work3/s204138/bach-models/trained_models/S3D_WLASL-91_epochs-3.358131_loss_0.300306_acc' #'WLASL/epoch299.pth.tar'
        self.backbone_weights_filename = 'KINETICS'
        self.head_weights_filename = None
        self.save_path = '/work3/s204138/bach-models/AblationS2G'
        self.checkpoint_path = None #'/work3/s204138/bach-models/Kinetics_CTC_training/S3D_PHOENIX-100_epochs-12.429813_loss_0.229911_WER' 
        ### for data augmentation ###
        self.crop_size = 224
        ### device ###
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_synthetic_glosses = True
        self.gloss_vocab, self.translation_vocab = getVocab('/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual', 
                                                            use_synthetic_glosses=self.use_synthetic_glosses)
        