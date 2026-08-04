import numpy as np
import os
import sys
import time
from . import util
from subprocess import Popen, PIPE
import numpy as np
from PIL import Image as PILImage
import wandb
import glob
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

import os
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_CONSOLE"] = "wrap"

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.wandb_project_name = opt.wandb_project_name
        self.current_epoch = 0
        self.train_mode = opt.train_mode
        self.raw_patch_size = opt.patch_size

        if self.use_wandb:
            self.wandb_run = wandb.init(dir=opt.checkpoints_dir, project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
            self.wandb_run._label(repo='CycleGAN-and-pix2pix')

        if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
            os.mkdir(os.path.join(opt.checkpoints_dir, opt.name))
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        if self.use_wandb:
            columns = [key for key, _ in visuals.items()]
            columns.insert(0, 'epoch')
            result_table = wandb.Table(columns=columns)
            table_row = [epoch]
            ims_dict = {}
            for label, image in visuals.items():

                if self.train_mode == '2d':
                    image_numpy = util.tensor2im(image)
                    if image_numpy.ndim >= 3:
                        wandb_image = wandb.Image(image_numpy[0])
                    else:
                        wandb_image = wandb.Image(image_numpy)
                elif self.train_mode == '3d':

                    #print(image.shape)


                    # if len(image.shape) == 6:
                    #     image_numpy = np.squeeze(util.tensor2im(image), axis=0)# N, c, H,W, D
                    #     image_numpy = np.squeeze(util.tensor2im(image_numpy), axis=0)
                    #     #print(image_numpy.shape)
                    #     image_numpy = image_numpy[:, :, self.raw_patch_size // 2]
                    #     image_numpy = (image_numpy * 255).astype(np.uint8)
                    #     #print(f"after uint8: {image_numpy.shape}")
                    #     if image_numpy.shape == (3, 256, 256):
                    #         image_numpy = np.transpose(image_numpy, (1, 2, 0))
                    #         pil_image = PILImage.fromarray(image_numpy, mode="RGB")
                    #         wandb_image = wandb.Image(pil_image)
                    #     else:
                    #         wandb_image = wandb.Image(image_numpy)
                    # else:
                    #     image_numpy = np.squeeze(util.tensor2im(image), axis=0)
                    #     image_numpy = image_numpy[:, :, self.raw_patch_size // 2]
                    #     image_numpy = (image_numpy * 255).astype(np.uint8)
                    #     wandb_image = wandb.Image(image_numpy)

                    if len(image.shape) == 5:
                        image_numpy = np.squeeze(util.tensor2im(image), axis=0)# N, c, H,W, D
                        #image_numpy = np.squeeze(util.tensor2im(image_numpy), axis=0)
                        #print(image_numpy.shape)
                        image_numpy = image_numpy[:, self.raw_patch_size // 2]
                        #image_numpy = (image_numpy * 255).astype(np.uint8)
                        #print(f"after uint8: {image_numpy.shape}")
                        if image_numpy.shape == (3, self.raw_patch_size, self.raw_patch_size):
                            image_numpy = (image_numpy * 255).astype(np.uint8)
                            image_numpy = np.transpose(image_numpy, (1, 2, 0))
                            pil_image = PILImage.fromarray(image_numpy, mode="RGB")
                            wandb_image = wandb.Image(pil_image)
                        else:
                            wandb_image = wandb.Image(image_numpy)
                    else:
                        image_numpy = util.tensor2im(image)
                        image_numpy = image_numpy[:, self.raw_patch_size // 2]
                        image_numpy = (image_numpy * 255).astype(np.uint8)
                        wandb_image = wandb.Image(image_numpy)

                table_row.append(wandb_image)
                ims_dict[label] = wandb_image
                media_dir = os.path.join(wandb.run.dir, "media/images")
                for f in glob.glob(os.path.join(media_dir, "*.png")):
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"Could not delete {f}: {e}")
            self.wandb_run.log(ims_dict)
            if epoch != self.current_epoch:
                self.current_epoch = epoch
                result_table.add_data(*table_row)
                self.wandb_run.log({"Result": result_table})


    def plot_current_losses(self, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if self.use_wandb:
            self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
