class cfg:
    def __init__(self) -> None:
        self.n_classes = 1085
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
        self.num_workers = 8
        self.print_freq = 50
        self.batch_size = 8
        self.backbone_weights_filename = 'WLASL/epoch299.pth.tar'
        self.head_weights_filename = None
        # verbose for weightloading #
        self.verbose = False


