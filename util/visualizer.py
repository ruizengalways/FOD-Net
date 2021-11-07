import os
from tensorboardX import SummaryWriter


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
        opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create an tensorboard file to store everything.
        """
        self.opt = opt  # cache the option
        self.name = opt.name

        # Create a tensorboard writer to store everything produced during training
        self.tensorboard_writer = SummaryWriter(
            os.path.join(opt.checkpoints_dir, opt.name, 'tfb_logs'))

    def plot_current_losses_tensorboard(self, losses, global_step):
        """display the current losses on tensorboardX

        Parameters:
            global_step (int)     -- current iteration
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for label, loss in losses.items():
            self.tensorboard_writer.add_scalar(tag=label,
                                               scalar_value=loss.item(),
                                               global_step=global_step)
