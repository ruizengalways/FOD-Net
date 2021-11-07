import argparse
import os
import torch
import models
import data
import numpy as np
import yaml
from easydict import EasyDict as edict
import re


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='fodnet', choices=['fodnet'],
                            help='chooses which model to use. [fodnet]')
        parser.add_argument('--init_type', type=str, default='xavier',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=1.,
                            help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='hcp',
                            help='chooses which dataset to use. This version only provide Human Connectome Project')
        parser.add_argument('--dataroot', type=str, help='path to DWI data (should have subfolders human no)')
        parser.add_argument('--num_threads', default=0, type=int, help='threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--cfg_file', default=None, type=str,
                            help='cfg file is used to override parameters specified in command line')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def cfg_parser(self, opt):
        """
        override options in the parser using the options specified in the cfg file
        :param opt: opt generated from the arguments
        :return: a cfg-modified opt
        """
        if opt.cfg_file is not None:
            print('Starting override args using the provided cfg file \n')
            with open(opt.cfg_file, 'r') as f:
                yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
            # Get available model options from cfg
            cfg_options_valid = np.zeros(len(yaml_cfg))
            for cfg_index, (cfg_name, cfg_value) in enumerate(yaml_cfg.items()):
                if cfg_name in vars(opt):
                    if isinstance(yaml_cfg[cfg_name], list):
                        setattr(opt, cfg_name, yaml_cfg[cfg_name])  # We assign a list of parameters to each scale
                    elif isinstance(getattr(opt, cfg_name), type(None)):
                        setattr(opt, cfg_name, yaml_cfg[cfg_name])
                    else:
                        setattr(opt, cfg_name, type(getattr(opt, cfg_name))(yaml_cfg[cfg_name]))
                    cfg_options_valid[cfg_index] = 1

            if np.any(cfg_options_valid == 0):
                print("The following options in the config file is invalid")
                for i, invaid_index in enumerate(list(np.where(cfg_options_valid == 0)[0].flatten())):
                    print('{0}. {1}: {2} is an invalid option '.format(i, list(yaml_cfg)[invaid_index],
                                                                       yaml_cfg[list(yaml_cfg)[invaid_index]]))
            # Check
            for opt_name, opt_value in vars(opt).items():
                if isinstance(opt_value, list):
                    assert (len(
                        opt_value) == opt.nscale), 'The length of the list should be equal to the number of scales'

        elif opt.cfg_file is None:
            print('There is no cfg file. Use the options specified in the command line. \n')

        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt = self.cfg_parser(opt)
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if opt.isTrain:
            self.print_options(opt)

        if opt.isTrain:
            if opt.continue_train:
                if opt.load_epoch == 'latest':
                    used_epochs = []
                    for i, file in enumerate(os.listdir(os.path.join(opt.checkpoints_dir, opt.name))):
                        regex = r"(\d{1,})\_net_(\S{1,})\.pth"
                        m = re.match(regex, file)
                        if m:
                            used_epochs.append(int(m.group(1)))
                    opt.epoch_count = np.max(used_epochs) + 1
                else:
                    opt.epoch_count = int(opt.load_epoch) + 1

                print('Re-starting from epoch %d' % opt.epoch_count)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
