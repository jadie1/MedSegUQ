import os
import yaml
import argparse
import importlib
import munch
import time
import json
import glob
import nrrd
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
    num_samples = int(args.num_samples)
    output_name = test_set + '_' + str(num_samples) + 'samples'
    if bool(args.test_time_dropout):
        output_name = output_name + '_ttd'

    train_files, val_files, test_files = get_file_sets(args)
    
    if test_set == 'train':
        test_data = train_files 
    elif test_set == 'val':
        test_data = val_files
    elif test_set == 'test':
        test_data = test_files
    elif test_set == 'ood1':
        data_dir = os.path.join(args.data_dir, args.dataset)
        test_images = sorted(glob.glob(os.path.join(data_dir, "imagesOOD1", "*.nii.gz")))
        test_labels = sorted(glob.glob(os.path.join(data_dir, "labelsOOD1", "*.nii.gz")))
        test_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
    elif test_set == 'ood2':
        data_dir = os.path.join(args.data_dir, args.dataset)
        test_images = sorted(glob.glob(os.path.join(data_dir, "imagesOOD2", "*.nii.gz")))
        test_labels = sorted(glob.glob(os.path.join(data_dir, "labelsOOD2", "*.nii.gz")))
        test_data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(test_images, test_labels)]
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

    if args.save_images:
        pred_dir = out_dir + output_name + '/predictions/'
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        unc_dir = out_dir  + output_name + '/uncertainty/'
        if not os.path.exists(unc_dir):
            os.makedirs(unc_dir)
    slice_dir = out_dir  + output_name + '/slices/'
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
    
    # device = 'cuda:2' # 
    device = args.device
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    roi_size = (96, 96, 96)
    if args.dataset == 'Task09_Spleen':
        sw_batch_size = 6
    elif args.dataset == 'Task07_Pancreas':
        sw_batch_size = 6



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
    inference_times, no_sample_dsc = [], []
    with Parallel(n_jobs=6) as parallel_backend:
        with torch.no_grad():
            for test_data in test_org_loader:
                image_name = os.path.basename(test_data['image_meta_dict']['filename_or_obj'][0])
                print(image_name)
                test_inputs = test_data["image"].to(device)
                start_time = time.time()
                test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
                inference_times.append(time.time()-start_time)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                test_pred, test_labels = from_engine(["pred", "label"])(test_data)
                dice_metric(y_pred=test_pred, y=test_labels)
                prediction = test_pred[0].cpu().numpy()
                seg = np.argmax(prediction, axis=0)
                gt = np.squeeze(test_labels[0].cpu().numpy())[1]
                no_sample_dsc += [dice_metric_numpy(ground_truth=gt, predictions=seg)]
            metric_org = dice_metric.aggregate().item()
            dice_metric.reset()
            inference_time = np.mean(np.asarray(inference_times))
            no_sample_dsc = np.asarray(no_sample_dsc) * 100.
    output['metrics']['no_sampling_DiceMetric'] = f"{metric_org:.2f}" 
    output['metrics']['no_sampling_inference_time'] = f"{inference_time:.4f}"
    output['metrics']['no_sampling_DSC'] = f"{np.mean(no_sample_dsc):.2f} +- {np.std(no_sample_dsc):.2f}"

    
    # Test UQ calibration (with dropout)
    if args.model_name == 'dropout' or args.model_name == 'concrete_dropout' or args.test_time_dropout:
        enable_dropout(model, output, bool(args.test_time_dropout))
    if 'members' in args:
        num_members = len(args.members)
    elif args.model_name == 'batch_ensemble':
        num_members = args.num_members
    else:
        num_members = 1
    # Parallelization is for numpy metric calculation
    with Parallel(n_jobs=6) as parallel_backend:
        dsc, dsc_aac, ndsc, f1, ndsc_aac, inference_times, total_unc, names = [], [], [], [], [], [], [], []
        with torch.no_grad():
            for i, test_data in enumerate(test_org_loader):
                image_name = os.path.basename(test_data['image_meta_dict']['filename_or_obj'][0])
                print(image_name)
                inputs = test_data["image"].to(device)
                all_outputs = []
                start_time = time.time()
                for samples_id in range(num_samples):
                    if args.model_name == 'swag':
                        model.sample()
                    test_data["pred"] = sliding_window_inference(inputs, roi_size, sw_batch_size, model, member_id=samples_id%num_members) # mode='gaussian'
                    # t_data = [post_transforms(i) for i in decollate_batch(test_data)]
                    # test_pred, labels = from_engine(["pred", "label"])(t_data)
                    # outputs = test_pred[0].cpu().numpy()
                    t_data = [post_transforms1(i) for i in decollate_batch(test_data)]
                    test_pred, labels = from_engine(["pred", "label"])(t_data)
                    probs = torch.nn.functional.softmax(test_pred[0], dim=0).cpu().numpy()
                    all_outputs.append(probs)
                inference_times.append(time.time()-start_time)
                all_outputs = np.asarray(all_outputs)

                # obtain binary segmentation mask
                prediction = np.mean(all_outputs, axis=0)
                seg = np.argmax(prediction, axis=0)

                gt = np.squeeze(labels[0].cpu().numpy())[1]

                # compute reverse mutual information uncertainty map
                unc_map = ensemble_uncertainties_classification(np.concatenate(
                    (np.expand_dims(all_outputs[:,1,:], axis=-1),
                     np.expand_dims(1. - all_outputs[:,1,:], axis=-1)),
                    axis=-1))['reverse_mutual_information']


                # Save slice plots
                slice_num = np.argmax(gt.sum(0).sum(0))
                plt.figure("check", (18, 12))
                plt.subplot(2, 3, 1)
                plt.title("Input image")
                test_image=nib.load(test_data['image_meta_dict']['filename_or_obj'][0]).get_fdata()
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
                if args.save_images:
                    # Save images
                    pred_file = os.path.join(pred_dir, image_name)
                    nrrd.write(pred_file.replace('.nii.gz','.nrrd'), seg)
                    unc_file = os.path.join(unc_dir, image_name)
                    nrrd.write(unc_file.replace('.nii.gz','.nrrd'), unc_map)

                # Compute metrics
                names.append(image_name.replace('.nii.gz',''))
                dsc += [dice_metric_numpy(ground_truth=gt, predictions=seg)]
                dsc_aac += [dsc_aac_metric(ground_truth=gt.flatten(),
                             predictions=seg.flatten(),
                             uncertainties=unc_map.flatten(),
                             parallel_backend=parallel_backend)]
                # f1 += [lesion_f1_score(ground_truth=gt,
                #                        predictions=seg,
                #                        IoU_threshold=0.5,
                #                        parallel_backend=parallel_backend)]
                ndsc += [dice_norm_metric(ground_truth=gt, predictions=seg)]
                ndsc_aac += [ndsc_aac_metric(ground_truth=gt.flatten(),
                                             predictions=seg.flatten(),
                                             uncertainties=unc_map.flatten(),
                                             parallel_backend=parallel_backend)]
                total_unc += [unc_map.sum()]

    dsc = np.asarray(dsc) * 100.
    dsc_aac = np.asarray(dsc_aac) * 100.
    # f1 = np.asarray(f1) * 100.
    ndsc = np.asarray(ndsc) * 100.
    ndsc_aac = np.asarray(ndsc_aac) * 100.
    inference_time = np.mean(np.asarray(inference_times))
    total_unc = np.asarray(total_unc)
    
    corr, _ = pearsonr(100-dsc, total_unc)
    ncorr, _ = pearsonr(100-ndsc, total_unc)
    output['values'] = {}
    output['values']['names'] = names
    output['values']['uncertainty'] = [str(x) for x in total_unc]
    output['values']['no_sample_DSC'] = [str(x) for x in no_sample_dsc]
    output['values']['DSC'] = [str(x) for x in dsc]
    output['values']['DSC_R-AUC'] = [str(x) for x in dsc_aac]
    output['values']['nDSC'] = [str(x) for x in ndsc]
    output['values']['nDSC_R-AUC'] = [str(x) for x in ndsc_aac]


    output['metrics']['sampling_inference_time'] = f"{inference_time:.4f}"
    output['metrics']['DSC'] = f"{np.mean(dsc):.2f} +- {np.std(dsc):.2f}"
    output['metrics']['R'] = f"{corr:.2f}"
    output['metrics']['DSC_R-AUC'] = f"{np.mean(dsc_aac):.2f} +- {np.std(dsc_aac):.2f}"
    output['metrics']['nDSC'] = f"{np.mean(ndsc):.2f} +- {np.std(ndsc):.2f}"
    output['metrics']['nR'] = f"{ncorr:.2f}"
    output['metrics']['nDSC_R-AUC'] = f"{np.mean(ndsc_aac):.2f} +- {np.std(ndsc_aac):.2f}"

    print("no sample DSC: ", f"{metric_org*100:.2f}")
    print(f"DSC:\t{np.mean(dsc):.2f} +- {np.std(dsc):.2f}")
    print(f"R:\t{corr:.2f}")
    print(f"DSC R-AUC:{np.mean(dsc_aac):.2f} +- {np.std(dsc_aac):.2f}")
    print(f"\nnDSC:\t{np.mean(ndsc):.2f} +- {np.std(ndsc):.2f}")
    print(f"nR:\t{ncorr:.2f}")
    print(f"nDSC R-AUC:{np.mean(ndsc_aac):.2f} +- {np.std(ndsc_aac):.2f}")

    # Save output
    json_object = json.dumps(output, indent=4)
    with open(out_dir + output_name + ".json", "w") as outfile:
        outfile.write(json_object)
                
def enable_dropout(model, output, test_time_dropout):
    if test_time_dropout:
        print("Using test time dropout.")
    count = 0
    for m in model.modules():
        if 'Dropout' in m.__class__.__name__:
            count += 1
            m.train()
            if test_time_dropout:
                m.p = 0.1
    # Ensemble
    if 'members' in args:
        for member in model.members:
            for m in member.modules():
                if 'Dropout' in m.__class__.__name__:
                    count += 1
                    m.train()
                    if test_time_dropout:
                        print("Dropout on.")
                        m.p = 0.1
    print("Turned on", count, "dropout layers.")

    # Print  learned concrete probabilities
    model_dict = model.state_dict()
    drop_keys = [key for key in model_dict.keys() if "p_logit" in key]
    if drop_keys:
        Ps = torch.empty(len(drop_keys))
        for i in range(len(drop_keys)):
            Ps[i] = torch.sigmoid(model_dict[drop_keys[i]])
        output['Dropout_probs'] = str(Ps.numpy())
        print("Dropout probs: ", Ps.numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-t', '--test_set', help='train, val, or test', default='test')
    parser.add_argument('-n', '--num_samples', help='number of samples', default=4)
    parser.add_argument('-ttd', '--test_time_dropout', help='Use MC dropopuot in testing', default=False)
    parser.add_argument('-s', '--save_images', help='If true predicted images will be saved.', default=False)
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
