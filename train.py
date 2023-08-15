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
from sklearn.model_selection import KFold
from torchinfo import summary
torch.autograd.set_detect_anomaly(True)

from monai.utils import first, set_determinism
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
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.apps import download_and_extract

from models.layers import ConcreteDropout
from models.layers import LP_BNN_Conv3d, LP_BNN_ConvTranspose3d
from models.layers import Rank1_BNN_Conv3d, Rank1_BNN_ConvTranspose3d


# from monai.config import print_config
# print_config()

def get_file_sets(args):
    ### Get Data
    # Setup data directory
    directory = args.data_dir
    root_dir = tempfile.mkdtemp() if directory is None else directory 
    # Download and extract dataset
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/"+args.dataset+".tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"
    compressed_file = os.path.join(root_dir, args.dataset+".tar")
    data_dir = os.path.join(root_dir, args.dataset)
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
    # Set data files
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    # Separate out test files using fold
    test_size = int(.2*len(data_dicts))
    val_size = int(.1*len(data_dicts))
    test_files = data_dicts[args.fold*test_size:((args.fold+1)*test_size)]
    train_val_files = [data_dict for data_dict in data_dicts if data_dict not in test_files]
    # Create validation split 
    val_fold = min(args.fold,3)
    val_files = train_val_files[val_fold*val_size:((val_fold+1)*val_size)]
    train_files = [data_dict for data_dict in train_val_files if data_dict not in val_files]
    return train_files, val_files, test_files


def train(args, log_dir):
    start_time = time.time()
    # Start logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    logging.info(str(args))

    # Create model, optimizer, and loss
    device = 'cuda:0' # torch.device(args.device) TODO
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    model = model_module.Model(args).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    if "CE_loss" in args:
            loss_function = torch.nn.CrossEntropyLoss()
    learning_rate = args.learning_rate if 'learning_rate' in args else 1e-4
    if 'weight_decay' in args:
        weight_decay=args.weight_decay
    else:
        weight_decay=0
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    if args.load_model:
        logging.info(" Loading %s model's weights.")
        model_dict = model.state_dict()
        intial_model_dict = torch.load(args.load_model, map_location=device)
        pretrained_dict = {k: v for k, v in intial_model_dict.items() if k in model_dict}
        not_pretrained_dict = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        logging.info(f'Initializing weights for: {pretrained_dict.keys()}. ')
        logging.info(f'No pretrained initialization for: {not_pretrained_dict.keys()}. ')
    # summary(model, input_size=(4, 1, 96, 96, 96)) 

    # Get data
    train_files, val_files, test_files = get_file_sets(args)

    # Get transforms
    train_transforms, val_transforms, val_org_transforms, post_transforms = get_transforms(args)
    # # Check transforms in dataloader
    # check_ds = Dataset(data=val_files, transform=val_transforms)
    # check_loader = DataLoader(check_ds, batch_size=1)
    # check_data = first(check_loader)
    # image, label = (check_data["image"][0][0], check_data["label"][0][0])
    # print(f"image shape: {image.shape}, label shape: {label.shape}")
    # # Plot check 
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # plt.imshow(image[:, :, 80], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(label[:, :, 80])
    # plt.show()


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


    # Set deterministic training for reproducibility
    set_determinism(seed=args.seed)

    # Train
    max_epochs = args.max_epochs
    val_interval = args.val_interval
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    early_stop_count = 0

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)

            reg = torch.zeros(1).to(device)
            for module in filter(lambda x: isinstance(x, ConcreteDropout), model.modules()):
                reg += module.regularization
            for module in filter(lambda x: isinstance(x, LP_BNN_ConvTranspose3d), model.modules()):
                reg += module.latent_loss
            for module in filter(lambda x: isinstance(x, LP_BNN_Conv3d), model.modules()):
                reg += module.latent_loss
            for module in filter(lambda x: isinstance(x, Rank1_BNN_ConvTranspose3d), model.modules()):
                reg += module.kld_loss
            for module in filter(lambda x: isinstance(x, Rank1_BNN_Conv3d), model.modules()):
                reg += module.kld_loss

            if "CE_loss" in args:
                loss = loss_function(outputs, labels.squeeze().long())
            else:
                loss = loss_function(outputs, labels) + reg



            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logging.info(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                    logging.info("saved new best metric model")
                    early_stop_count = 0
                else:
                    if epoch > 50: # Small lag for lp-bnn
                        early_stop_count += 1
                logging.info(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                    f"\nearly stop count: {early_stop_count}"
                )

        if early_stop_count > args.early_stop_patience:
            print("Early stopping epoch:", epoch)
            break
        logging.info(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")
    logging.info(f"Time: {time.time()-start_time:.4f} ")


    # Plot the loss and metric
    plot_loss_and_metric(epoch_loss_values, metric_values, log_dir, val_interval)

    # Plot val examples
    plot_examples(model, log_dir, val_loader, device)
   
    # Eval on original image spacings
    val_org_ds = Dataset(data=val_files, transform=val_org_transforms)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=args.num_workers)
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        for val_data in val_org_loader:
            torch.cuda.empty_cache()
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
    logging.info(f"Metric on original image spacing: {metric_org:.4f} ")


def plot_loss_and_metric(epoch_loss_values, metric_values, log_dir, val_interval):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(log_dir + '/train_plot.png')
    plt.clf()

def plot_examples(model, log_dir, val_loader, device, num_examples=3, name='val'):
    # Check best model output with the input image and label
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pth")))
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data["image"].to(device), roi_size, sw_batch_size, model)
            slice_num = 80 # val_data["image"].shape[-1]//2
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(val_data["image"][0, 0, :, :, slice_num], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(val_data["label"][0, 0, :, :, slice_num])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_num])
            plt.savefig(log_dir+'/'+name+'_output'+str(i)+'.png')
            plt.clf()
            if i == (num_examples-1):
                break
def get_transforms(args):
    # Setup transforms for training and validation
    #   Here we use several transforms to augment the dataset:
    #   1. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
    #   2. `EnsureChannelFirstd` ensures the original data to construct "channel first" shape.
    #   3. `Orientationd` unifies the data orientation based on the affine matrix.
    #   4. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
    #   5. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
    #   6. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
    #   7. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.  
    #   The image centers of negative samples must be in valid body area.
    #   8. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
    if args.dataset == 'Task09_Spleen':
        a_min = -57
        a_max = 164
        pixdim = (1.5, 1.5, 2.0)
    elif args.dataset == 'Task07_Pancreas':
        a_min = -87
        a_max = 199
        pixdim = (1.5, 1.5, 1.0)
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=a_min,
                a_max=a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ]
    )
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # CropForegroundd(keys=["image"], source_key="image"),
        ]
    )
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=val_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            AsDiscreted(keys="label", to_onehot=2),
        ]
    )
    return train_transforms, val_transforms, val_org_transforms, post_transforms


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

    train(args, log_dir)

    args['best_model_path'] = os.path.join(log_dir, "best_model.pth")
    # Update yaml in log dir
    with open(os.path.join(log_dir, os.path.basename(config_path)), 'w') as f:
        yaml.dump(args, f)
    print(os.path.join(log_dir, os.path.basename(config_path)))

    # Test
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path)), '-d', args.device])
    subprocess.call(['python', 'test.py', '-c', os.path.join(log_dir, os.path.basename(config_path)), '-n', str(30), '-d', args.device])
