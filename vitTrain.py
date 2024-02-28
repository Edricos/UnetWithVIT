import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm
import dataset_util as dsutil
import model_io
import VIT
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize


class TrainingConfig:
    def __init__(self):
        self.epochs = 25  # Number of total epochs to run
        self.n_bins = 80  # Number of bins/buckets to divide depth range into
        # self.lr = 0.000357  # Max learning rate
        self.lr = 0.00001  # Max learning rate
        self.wd = 0.1  # Weight decay
        self.w_chamfer = 0.1  # Weight value for chamfer loss
        self.div_factor = 25  # Initial div factor for lr
        self.final_div_factor = 100  # Final div factor for lr
        self.bs = 16  # Batch size
        self.validate_every = 100  # Validation period
        self.gpu = None  # Which GPU to use
        self.name = "UnetAdaptiveBins"  # Default model name
        self.norm = "linear"  # Type of norm/competition for bin-widths
        self.same_lr = False  # Use same LR for all param groups
        self.distributed = True  # Use DDP if set
        self.root = "."  # Root folder to save data in
        self.resume = ''  # Resume from checkpoint
        self.notes = ''  # Wandb notes
        self.tags = 'sweep'  # Wandb tags
        self.workers = 11  # Number of workers for data loading
        self.dataset = 'nyu'  # Dataset to train on
        self.data_path = '../dataset/nyu/sync/'  # Path to dataset
        self.gt_path = '../dataset/nyu/sync/'  # Path to dataset ground truth
        self.filenames_file = "./train_test_inputs/nyudepthv2_train_files_with_gt.txt"  # Path to the filenames text file
        self.input_height = 416  # Input height
        self.input_width = 544  # Input width
        self.max_depth = 10  # Maximum depth in estimation
        self.min_depth = 1e-3  # Minimum depth in estimation
        self.do_random_rotate = True  # If set, will perform random rotation for augmentation
        self.degree = 2.5  # Random rotation maximum degree
        self.do_kb_crop = False  # If set, crop input images as KITTI benchmark images
        self.use_right = False  # If set, will randomly use right images when train on KITTI
        self.data_path_eval = "../dataset/nyu/official_splits/test/"
        # Path to the data for online evaluation
        self.gt_path_eval = "../dataset/nyu/official_splits/test/"
        # Path to the ground truth data for online evaluation
        self.filenames_file_eval = "./train_test_inputs/nyudepthv2_test_files_with_gt.txt"
        # Path to the filenames text file for online evaluation
        self.min_depth_eval = 1e-3  # Minimum depth for evaluation
        self.max_depth_eval = 10  # Maximum depth for evaluation
        self.eigen_crop = True  # If set, crops according to Eigen NIPS14
        self.garg_crop = False  # If set, crops according to Garg ECCV16
        self.epoch = 0
        self.last_epoch = -1



def is_rank_zero(args):
    return args.rank == 0

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines




def validate(args, model, criterion_ueff, epoch, epochs, device='cpu'):
    data_fpath = 'splits/'
    # 数据加载
    fpath = os.path.join(data_fpath, "real_{}_night.txt")
    train_filenames = readlines(fpath.format("test"))

    max_distance = 150
    min_distance = 3
    img_id = {}
    base_dir = '/home/edric/PycharmProjects/UnetWithVIT/data/real'
    Results = []
    gta_pass = ''

    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for i in tqdm(range(0, len(train_filenames)), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"):
            # img = batch['image'].to(device)
            # img = batch['image'].to(device)
            img_id[i] = train_filenames[i].split('\n')
            id = img_id[i][0]
            gate_dir = os.path.join(base_dir, 'gated{}_10bit', '{}.png'.format(id))

            in_img = dsutil.read_gated_image(base_dir=base_dir, gta_pass=gta_pass, img_id=id, data_type='real')
            in_img = torch.tensor(in_img).to(device=device)
            img = in_img.permute(0, 3, 1, 2)

            input, lidar_mask = dsutil.read_gt_image(base_dir=base_dir, gta_pass=gta_pass, img_id=id,
                                                     data_type='real', min_distance=min_distance,
                                                     max_distance=max_distance)
            input = torch.tensor(input).to(device=device)
            depth = input.permute(0, 3, 1, 2)
            # depth = batch['depth'].to(device)
            #             # if 'has_valid_depth' in batch:
            #             #     if not batch['has_valid_depth']:
            #             #         continue


            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            # mask = depth > args.min_depth
            mask = torch.tensor(lidar_mask).permute(0, 3, 1, 2)
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            # pred[pred < args.min_depth_eval] = args.min_depth_eval
            # pred[pred > args.max_depth_eval] = args.max_depth_eval
            # pred[np.isinf(pred)] = args.max_depth_eval
            # pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            # valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            # if args.garg_crop or args.eigen_crop:
            #     gt_height, gt_width = gt_depth.shape
            #     eval_mask = np.zeros(valid_mask.shape)
            #
            #     if args.garg_crop:
            #         eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
            #         int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            #
            #     elif args.eigen_crop:
            #         if args.dataset == 'kitti':
            #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
            #             int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            #         else:
            #             eval_mask[45:471, 41:601] = 1
            # valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[mask.squeeze()], pred[mask.squeeze()]))

        return metrics.get_value(), val_si


def train():
    # Create an instance of TrainingConfig
    args = TrainingConfig()
    print(args.epochs)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)



    model = VIT.UnetAdaptiveBins.build(
        n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm)
    model.to(device=device)



    epochs=10
    experiment_name="DeepLab"
    lr=0.0001
    root="."
    optimizer_state_dict=None

    epochs=args.epochs
    experiment_name=args.name
    lr=args.lr
    root=args.root


    global PROJECT

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss()
    ################################################################################################

    model.train()


    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m =  model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################

    data_fpath = 'splits/data'
    # 数据加载
    fpath = os.path.join(data_fpath, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))

    max_distance = 150
    min_distance = 3
    img_id = {}
    base_dir = '/home/edric/PycharmProjects/UnetWithVIT/data/real'
    gta_pass = ''

    # some globals
    iters = len(train_filenames)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_filenames),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################




    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        # if should_log: wandb.log({"Epoch": epoch}, step=step)
        print("Epoch", epoch)
        for i, batch in tqdm(enumerate(range(0, len(train_filenames))),
                             desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train"):
            # break

            optimizer.zero_grad()

            # img = batch['image'].to(device)
            img_id[i] = train_filenames[i].split('\n')
            id = img_id[i][0]
            gate_dir = os.path.join(base_dir, 'gated{}_10bit', '{}.png'.format(id))

            in_img = dsutil.read_gated_image(base_dir=base_dir, gta_pass=gta_pass, img_id=id, data_type='real')
            in_img = torch.tensor(in_img).to(device=device)
            img = in_img.permute(0, 3, 1, 2)

            # depth = batch['depth'].to(device)
            input, lidar_mask = dsutil.read_gt_image(base_dir=base_dir, gta_pass=gta_pass, img_id=id,
                                                     data_type='real', min_distance=min_distance,
                                                     max_distance=max_distance)
            input = torch.tensor(input).to(device=device)
            depth = input.permute(0, 3, 1, 2)

            # if 'has_valid_depth' in batch:
            #     if not batch['has_valid_depth']:
            #         continue

            bin_edges, pred = model(img)

            # mask = depth > args.min_depth
            mask = torch.tensor(lidar_mask).permute(0, 3, 1, 2)
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            # if step % 5 == 0:
                # print('   criterion_ueff: ', '{:.4f}'.format(l_dense.item()),
                #       '   criterion_bins: ', '{:.4f}'.format(l_chamfer.item()),
                #       '   loss: ', '{:.4f}'.format(loss.item()))

                # wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                # wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)


            step += 1
            scheduler.step()

            ########################################################################################################
            ################################# Validation loop ##################################################
        model.eval()
        metrics, val_si = validate(args, model, criterion_ueff, epoch, epochs, device)
        print("val_si: {}".format(val_si))
        print("Validated: {}".format(metrics))
            #
            # if should_write and step % args.validate_every == 0:
                ################################# Validation loop ##################################################
                # model.eval()
                # metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)
                #
                # # print("Validated: {}".format(metrics))
                # if should_log:
                #     wandb.log({
                #         f"Test/{criterion_ueff.name}": val_si.get_value(),
                #         # f"Test/{criterion_bins.name}": val_bins.get_value()
                #     }, step=step)
                #
                #     wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                #     model_io.save_checkpoint(model, optimizer, epoch,
                #                              f"{experiment_name}_{run_id}_latest.pt",
                #                              root=os.path.join(root, "checkpoints"))
                #
                # if metrics['abs_rel'] < best_loss and should_write:
                #     model_io.save_checkpoint(model, optimizer, epoch,
                #                              f"{experiment_name}_{run_id}_best.pt",
                #                              root=os.path.join(root, "checkpoints"))
                #     best_loss = metrics['abs_rel']
                # model.train()
                #################################################################################################
        model.train()


train()

