from .base_options import BaseOptions


def nullable_string(val):
    if not val:
        return None
    return val


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--fod_path', type=str, help="The path to the fod image")
        parser.add_argument('--output_path', type=str, help="The output path")
        parser.add_argument('--weights_path', type=str, default='placeholder')
        parser.add_argument('--brain_mask_path', type=str, default='placeholder')
        self.isTrain = False
        return parser
