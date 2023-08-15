import os
import yaml
import argparse
import importlib
import munch
import time
import json
import glob
import numpy as np
import nibabel as nib
from joblib import Parallel
import torch
import torch.optim as optim
import warnings
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from warnings import filterwarnings
filterwarnings(action='ignore', message='In the future `np.bool` will be defined as the corresponding NumPy scalar.')

from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
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

from uncertainty import ensemble_uncertainties_classification
from metrics import dice_norm_metric, lesion_f1_score, ndsc_aac_metric, dice_metric_numpy, dsc_aac_metric
from train import plot_examples, get_file_sets, get_transforms


def test(test_set, args):
    # create output dict
    out_dir = os.path.dirname(config_path) + '/'
    output = {'config':args, 'metrics':{}}
    output_name = test_set + '_' + 'base'

    train_files, val_files, test_files = get_file_sets(args)
    
    if test_set == 'train':
        test_data = train_files 
    elif test_set == 'val':
        test_data = val_files
    elif test_set == 'test':
        test_data = test_files
    elif test_set == 'ood':
        data_dir = os.path.join(args.data_dir, args.dataset)
        test_images = sorted(glob.glob(os.path.join(data_dir, "imagesOOD", "*.nii.gz")))
        test_labels = sorted(glob.glob(os.path.join(data_dir, "labelsOOD", "*.nii.gz")))
        test_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]

    train_transforms, _, test_org_transforms, post_transforms = get_transforms(args)
    post_transforms1 = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            AsDiscreted(keys="label", to_onehot=2),
        ]
    )
    post_transforms2 = Compose(
        [
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        ]
    )

    # # for updating swag batch norm
    # if args.model_name == 'swag':
    #     train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=args.num_workers)
    #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=0) # num_workers=4 

    slice_dir = out_dir  + output_name + '/slices/'
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
    
    # device = 'cuda:2' # 
    device = args.device
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    roi_size = (96, 96, 96)
    if args.dataset == 'Task09_Spleen':
        sw_batch_size = 4
    elif args.dataset == 'Task07_Pancreas':
        sw_batch_size = 4


    # Load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    model = model_module.Model(args).to(device)
    if args.model_name != 'ensemble' and args.model_name != 'multi_swag':
        model.load_state_dict(torch.load(args.best_model_path,map_location=device))
    print("%s model's previous weights loaded." % args.model_name)
    # summary(model, [(1,96,96,96)])
    if args.model_name == 'swag':
        model.subspace.rank = torch.tensor(0)
    model.eval()

    # Test prediction accuracy (no dropout, avg of ensembles)
    inference_times, no_sample_dsc, total_unc, ns_dsc_aac, names = [], [], [], [], []
    with Parallel(n_jobs=6) as parallel_backend:
        with torch.no_grad():
            for test_data in test_org_loader:
                test_image=nib.load(test_data['image_meta_dict']['filename_or_obj'][0]).get_fdata()
                image_name = os.path.basename(test_data['image_meta_dict']['filename_or_obj'][0])
                print(image_name)
                names.append(image_name.replace('.nii.gz',''))
                test_inputs = test_data["image"].to(device)
                start_time = time.time()
                test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
                inference_times.append(time.time()-start_time)
                test_data = [post_transforms1(i) for i in decollate_batch(test_data)]
                test_pred, test_labels = from_engine(["pred", "label"])(test_data)
                probs = torch.nn.functional.softmax(test_pred[0], dim=0).cpu().numpy()
                unc_map = 1-np.max(probs, axis=0)
                test_data = post_transforms2(test_data)
                test_pred, test_labels = from_engine(["pred", "label"])(test_data)
                dice_metric(y_pred=test_pred, y=test_labels)
                seg = np.argmax(probs, axis=0)
                total_unc += [unc_map.sum()]
                gt = np.squeeze(test_labels[0].cpu().numpy())[1]
                no_sample_dsc += [dice_metric_numpy(ground_truth=gt, predictions=seg)]
                ns_dsc_aac += [dsc_aac_metric(ground_truth=gt.flatten(),
                             predictions=seg.flatten(),
                             uncertainties=unc_map.flatten(),
                             parallel_backend=parallel_backend)]
                # Save slice plots
                slice_num = np.argmax(gt.sum(0).sum(0))
                plt.figure("check", (18, 12))
                plt.subplot(2, 3, 1)
                plt.title("Input image")
                test_image = (test_image-np.min(test_image))/(np.max(test_image)-np.min(test_image))
                plt.imshow(test_image[:, :, slice_num], cmap="gray")
                plt.subplot(2, 3, 2)
                plt.title("Label")
                plt.imshow(gt[ :, :, slice_num], cmap="gray")
                plt.subplot(2, 3, 3)
                plt.title("Prediction")
                plt.imshow(seg[ :, :, slice_num], cmap="gray")
                plt.subplot(2, 3, 5)
                plt.title("Error")
                plt.imshow(seg[ :, :, slice_num], cmap="gray")
                # get colormap
                c_white = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
                c_red = matplotlib.colors.colorConverter.to_rgba('red',alpha = 1)
                cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)
                plt.imshow(np.abs(gt[:,:,slice_num]-seg[:, :, slice_num]), cmap=cmap_rb, interpolation=None, alpha=1)
                plt.subplot(2, 3, 6)
                plt.title("Predicted uncertainty")
                plt.imshow(unc_map[:, :, slice_num], cmap='Reds')
                plt.savefig(slice_dir+image_name.replace('.nii.gz','.png'))
                plt.clf()
            metric_org = dice_metric.aggregate().item()
            dice_metric.reset()
            inference_time = np.mean(np.asarray(inference_times))
            no_sample_dsc = np.asarray(no_sample_dsc) * 100.
            ns_dsc_aac = np.asarray(ns_dsc_aac) * 100.
    total_unc = np.asarray(total_unc)
    ns_corr, _ = pearsonr(no_sample_dsc, total_unc)
    output['metrics']['DiceMetric'] = f"{metric_org:.2f}" 
    output['metrics']['inference_time'] = f"{inference_time:.4f}"
    output['metrics']['DSC'] = f"{np.mean(no_sample_dsc):.2f} +- {np.std(no_sample_dsc):.2f}"
    output['metrics']['R'] = f"{ns_corr:.2f}"
    output['metrics']['DSC_R-AUC'] = f"{np.mean(ns_dsc_aac):.2f} +- {np.std(ns_dsc_aac):.2f}"
    output['values'] = {}
    output['values']['names'] = names
    output['values']['uncertainty'] = [str(x) for x in total_unc]
    output['values']['DSC'] = [str(x) for x in no_sample_dsc]
    output['values']['DSC_R-AUC'] = [str(x) for x in ns_dsc_aac]

    print("DSC: ", f"{metric_org*100:.2f}")
    print(f"R:\t{ns_corr:.2f}")
    print(f"DSC R-AUC:{np.mean(ns_dsc_aac):.2f} +- {np.std(ns_dsc_aac):.2f}")

    # Save output
    json_object = json.dumps(output, indent=4)
    with open(out_dir + output_name + ".json", "w") as outfile:
        outfile.write(json_object)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-t', '--test_set', help='train, val, or test', default='test')
    parser.add_argument('-d', '--device', default='cuda:0')
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    args.update(arg.__dict__)
   
    if arg.test_set == 'all':
        for test_set in ['train', 'val', 'test']:
            test(test_set, args)
    else:
        test(arg.test_set, args)
