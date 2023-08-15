import os
import sys
import yaml
import argparse
import logging
import math
import importlib
import datetime
import random
import munch
import time
import torch
import torch.optim as optim
import warnings
import shutil
import subprocess
import matplotlib.pyplot as plt
import tempfile
import glob
import tqdm
import itertools
from sklearn.model_selection import KFold
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)

from monai.utils import first, set_determinism
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.apps import download_and_extract
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)

from models.layers import ConcreteDropout
from train import get_transforms, plot_examples, plot_loss_and_metric, get_file_sets

# from monai.config import print_config
# print_config()

def train(args, log_dir):
    # Start logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    logging.info(str(args))

    ### Get Data
    train_files, val_files, test_files = get_file_sets(args)

    # Create model, optimizer, and loss
    device = torch.device(args.device)
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    model = model_module.Model(args).to(device)
    if device == 'cuda:0':
        summary(model, [(1,96,96,96)]) # requires device to be cuda:0
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Load base model
    model_module = importlib.import_module('.%s' % args.base_model_name, 'models')
    base_model = model_module.Model(args).to(device)
    base_model.load_state_dict(torch.load(args.base_model_path, map_location=device))
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    learning_rate = args.learning_rate if 'learning_rate' in args else 1e-4
    optimizer = torch.optim.Adam(base_model.parameters(), learning_rate)
    base_dice_metric = DiceMetric(include_background=False, reduction="mean")


    # Set deterministic training for reproducibility
    set_determinism(seed=args.seed)

    # Get transforms
    train_transforms, val_transforms, val_org_transforms, post_transforms = get_transforms(args)

    # Create data loaders
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
    # train_ds = Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)
    logging.info('Length of train dataset:%d', len(train_ds))
    logging.info('Length of val dataset:%d', len(val_ds))


    # Train
    max_epochs = args.swa_epochs
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    early_stop_count = 0

    for epoch in range(max_epochs):
        # Train base model 
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        base_model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = base_model(inputs)

            # For concrete dropout training
            reg = torch.zeros(1).to(device)
            for module in filter(lambda x: isinstance(x, ConcreteDropout), base_model.modules()):
                reg += module.regularization

            loss = loss_function(outputs, labels) + reg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logging.info(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"SWA epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        model.collect_model(base_model)

        if (epoch + 1) % val_interval == 0:
            model.set_swa()
            bn_update(train_loader, model)
            # TODO BN update
            model.eval()
            base_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    # compute SWA metric for current iteration
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    # Compute base metric for current iteration 
                    base_val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, base_model)
                    base_val_outputs = [post_pred(i) for i in decollate_batch(base_val_outputs)]
                    base_dice_metric(y_pred=base_val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                base_metric = base_dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                base_dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                logging.info("saved new best metric model")
                logging.info(
                    f"current epoch: {epoch + 1} current mean SWA dice: {metric:.4f}"
                    f"\ncurrent mean base dice: {base_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        logging.info(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")


    # Plot the loss and metric
    plot_loss_and_metric(epoch_loss_values, metric_values, log_dir, val_interval)

    # Plot val examples
    plot_examples(model, log_dir, val_loader, device)
   
    # Eval on original image spacings
    val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=args.num_workers)
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))
    model.set_swa()
    bn_update(train_loader, model)
    model.eval()
    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
        # aggregate the final mean dice result
        metric_org = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
    print("Metric on original image spacing: ", metric_org)


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)

        for data in loader:
            input = data["image"].to(model.device)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    print_time = datetime.datetime.now().isoformat()[:19]
    exp_name = args.model_name 
    # exp_name += '_'+print_time.replace(':',"-")
    if 'base_model_name' in args:
        exp_name += '_'+args.base_model_name
    exp_name += '_seed'+str(args.seed)
    log_dir = os.path.join(args.work_dir, args.dataset, 'fold'+str(args.fold), exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(log_dir)

    if 'base_model_path' not in args:
        print("Error: Trained base model path must be provided to train swag.")
    if 'base_model_name' not in args:
        print("Error: Name of base model must be provided to train swag.")

    train(args, log_dir)

    args['best_model_path'] = os.path.join(log_dir, "best_model.pth")
    # Update yaml in log dir
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)
    print(os.path.join(log_dir, os.path.basename(config_path)))

     # Test
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path)), '-d', args.device])
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path)), '-n', str(30), '-d', args.device])
