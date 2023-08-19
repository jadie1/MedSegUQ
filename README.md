# MedSegUQ
This repo contains the source code for the paper ["Benchmarking Scalable Epistemic Uncertainty Quantification in Organ Segmentation"](https://arxiv.org/abs/2308.07506). To be presented at the [UNSURE workshop](https://unsuremiccai.github.io/) held in conjunction with MICCAI 2023. 

Code dependencies can be found in `requirements.txt`

Models are defined via a `.yaml` configuration file. Example configuration files are provided in `cfgs/`.

## Model Training
To train a model, call `train.py` with the configuration file. For example:
```
python train.py -c cfgs/base.yaml
```
This will download the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset specified in the config file and train the model. 

If training a `swag` model, train a `base` model first, then use `train_swag.py`. For example:
```
python train_swag.py -c cfgs/swag.yaml
```

If using an `ensemble` model, train each member separately, then add them to a shared config file, such as in `cfgs/ensemble_base.yaml` for testing. 

## Model Testing
When training is complete, the `.yaml` file will be copied to the appropriate folder in the specified `work_dir` and the `best_model_path` will be added to it.

To test a `base` model, call `test_base.py` with the updated config file and the set to test (train, val, test, or all). For example:
```
python test_base.py -c experiments/Task09_Spleen/fold0/base_seed0/base.yaml -t test
```

To test any other model, call `test.py` with the config file, the set to test, and the number of samples to use in uncertainty quantification. For example: 
```
python test.py -c experiments/Task09_Spleen/fold0/base_seed0/concrete_dropout.yaml -t test -n 4
```
If `-ttd True` is used, test time dropout will be used in prediction.
If `-s True` is used, the predicted segmentations and uncertainty maps will be written to `.nrrd` files.

When `test_base.py` and `test.py` are called, the results are printed and written to a `.json` file. Visualization of slices is also saved. 

## References

Code is based on and uses the [MONAI](https://monai.io/) toolkit:

- [MONAI Spleen Segmentation](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb)
- [MONAI Residual UNet](https://docs.monai.io/en/stable/_modules/monai/networks/nets/unet.html) based on ["Left-Ventricle Quantification Using Residual U-Net"](https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40)

Evaluation code was adapted from the "Multiple Sclerosis White Matter Lesions Segmentation" [Shifts Challenge](https://github.com/Shifts-Project/):

- [metrics.py](https://github.com/Shifts-Project/shifts/blob/main/mswml/metrics.py)
- [uncertainty.py](https://github.com/Shifts-Project/shifts/blob/main/mswml/uncertainty.py)

The code for each epistemic uncertainty quantification method is based on the following repos:

- Ensemble: [Paper](https://arxiv.org/abs/1612.01474)
- Batch Ensemble: [Paper](https://arxiv.org/abs/2002.06715), [Code](https://github.com/giannifranchi/LP_BNN)
- MC Dropout: [Paper](https://arxiv.org/abs/1506.02142)
- Concrete Dropout: [Paper](https://arxiv.org/abs/1705.07832), [Code](https://github.com/yaringal/ConcreteDropout)
- Rank1 BNN: [Paper](https://arxiv.org/pdf/2005.07186.pdf), [Code](https://github.com/google/edward2)
- LP-BNN: [Paper](https://arxiv.org/abs/2012.02818), [Code](https://github.com/giannifranchi/LP_BNN)
- SWAG: [Paper](https://arxiv.org/pdf/1902.02476.pdf), [Code](https://github.com/izmailovpavel/understandingbdl)
- Multi-SWAG: [Paper](https://arxiv.org/abs/2002.08791), [Code](https://github.com/izmailovpavel/understandingbdl)
