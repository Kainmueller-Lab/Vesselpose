import tifffile
import torch
from data import create_dataset
from torch.utils.data import DataLoader
from argparse import Namespace
from util.visualizer import Visualizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import BCEWithLogitsLoss, MSELoss
from funlib.learn.torch.models import UNet, ConvPass

import os
import matplotlib.pyplot as plt
from copy import deepcopy
from config_utils import load_config

def train(opt):
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     dataset = create_dataset(opt)
     dataset_size = len(dataset)
     print('The number of training images = %d' % dataset_size)
     print(len(dataset))
     if len(dataset) == 0:
          raise ValueError("Dataset is empty. Please check your data loading implementation.")

     out_channel = 4  # fg + 3 dim direction vector
     downsample_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
     ks = [3] * 3
     ks = [ks] * 2
     ks = [ks] * (len(downsample_factors) + 1)
     print(ks)
     unet = UNet(
         in_channels=1,
         num_fmaps=16,
         fmap_inc_factor=2,
         # fmap_dec_factor=[2, 2, 2, 2],
         downsample_factors=downsample_factors,
         kernel_size_down=ks,
         kernel_size_up=ks,
         # upsampling="transposed_conv",
         padding="same",
         num_heads=1
     ).to(device)
     print("unet created")

     if opt.last_conv_batchnorm:
          last_conv = torch.nn.Sequential(
               ConvPass(unet.out_channels, out_channel, [[1, 1, 1]], activation=None, padding='same'),
               torch.nn.BatchNorm3d(out_channel)
          )
     else:
          last_conv = ConvPass(
               unet.out_channels, out_channel, [[1, 1, 1]], activation=None, padding='same'
          )


     print("last conv created")
     model = torch.nn.Sequential(
         unet,
         last_conv
     )
     print("model created")
     model = model.to(device)
     print("model to device")

     optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
     max_epochs = opt.max_epochs
     print(f"number of epochs:{max_epochs}")

     def linear_decay(epoch):
          return 1 - epoch / max_epochs

     visualizer = Visualizer(opt)
     scheduler = LambdaLR(optimizer, lr_lambda=linear_decay)


     train_loader = dataset
     val_loader = deepcopy(dataset)

     train_losses = []
     val_losses = []

     for epoch in range(max_epochs):
          print(f"Starting epoch {epoch + 1}")
          model.train()
          epoch_loss = 0.0
          visualizer.reset()
          print("Visualizer is reset")
          if hasattr(train_loader.dataset, 'shuffle_patch_pairs'):
               train_loader.dataset.shuffle_patch_pairs()


          for batch_idx, (images, labels) in enumerate(train_loader):
               images, labels = images.to(device), labels.to(device)

               seg_mask = labels[:, 0:1, ...]

               ground_truth_foreground_mask = (seg_mask > 0.5).float()
               weight_map = ground_truth_foreground_mask * 1.0 + (1 - ground_truth_foreground_mask) * 0.1
               direction_vectors = labels[:, 1:, ...]

               optimizer.zero_grad()
               outputs = model(images)

               predicted_seg_mask = outputs[:, 0:1, ...]

               predicted_direction_vectors = outputs[:, 1:, ...]

               if opt.bce_pos_weight is not None:
                    pos_weight = torch.tensor([opt.bce_pos_weight]).to(device)
                    BCEloss = BCEWithLogitsLoss(pos_weight=pos_weight)(predicted_seg_mask, seg_mask)
               else:
                    BCEloss = BCEWithLogitsLoss()(predicted_seg_mask, seg_mask)

               mseloss_vectors = (weight_map * (predicted_direction_vectors - direction_vectors) ** 2).mean()
               lambda_seg = 1
               lambda_vec = 1
               loss = lambda_seg * BCEloss + lambda_vec * mseloss_vectors

               loss.backward()
               optimizer.step()

               epoch_loss += loss.item()

               predictions = (torch.sigmoid(predicted_seg_mask) > 0.5).float()

               predictions_vector = predicted_direction_vectors > 0

               visuals = {'output image': predictions, 'flow_vectors': predictions_vector, 'input image': images,
                          'Labels': seg_mask}
               losses = {'Segmentation(BCE)': BCEloss, 'vectors(MSE)': mseloss_vectors}
               visualizer.display_current_results(visuals, epoch)
               visualizer.plot_current_losses(losses)


          avg_train_loss = epoch_loss / len(train_loader)
          train_losses.append(avg_train_loss)


          scheduler.step()


          model.eval()
          with torch.no_grad():
               val_loss = 0.0
               for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)

                    seg_mask = val_labels[:, 0:1, ...]
                    ground_truth_foreground_mask = (seg_mask > 0.5).float()
                    weight_map = ground_truth_foreground_mask * 1.0 + (1 - ground_truth_foreground_mask) * 0.1
                    direction_vectors = val_labels[:, 1:, ...]

                    val_outputs = model(val_images)

                    val_predicted_seg_mask = val_outputs[:, 0:1, ...]
                    val_predicted_direction_vectors = val_outputs[:, 1:, ...]

                    val_BCEloss = BCEWithLogitsLoss()(val_predicted_seg_mask, seg_mask)
                    val_mseloss_vectors = (
                                 weight_map * (val_predicted_direction_vectors - direction_vectors) ** 2).mean()
                    lambda_seg = 1
                    lambda_vec = 1

                    val_loss = lambda_seg * val_BCEloss + lambda_vec * val_mseloss_vectors


               avg_val_loss = val_loss / len(val_loader)
               val_losses.append(avg_val_loss.item())

               print(f"Validation Loss: {avg_val_loss.item()}")

          print(f"Epoch {epoch + 1}/{max_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss.item()}")

          checkpoints_dir = opt.checkpoints_dir

          if (epoch + 1) % 5 == 0:
               checkpoint_path = os.path.join(checkpoints_dir, f'unet_epoch_{epoch + 1}.pth')
               torch.save(model.state_dict(), checkpoint_path)
               print(f"Checkpoint saved at {checkpoint_path}")


     torch.save(model.state_dict(), "unet_segmentation_model.pth")


opt = load_config("config.yaml")
train(opt)