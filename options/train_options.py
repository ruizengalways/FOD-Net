from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # Continue training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')

        # training parameters
        parser.add_argument('--epoch_count', type=int, default=1, help='Which epoch to start the training process')
        parser.add_argument('--epoch_end', type=int, default=50000, help='Which epoch to terminate the training process')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # Export results to tensorboard
        parser.add_argument('--print_freq', type=int, default=5, help='frequency of showing training results on console')

        self.isTrain = True
        return parser
