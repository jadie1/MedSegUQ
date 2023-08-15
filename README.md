# MedSegUQ
This repo contains the source code for the paper "Benchmarking Scalable Epistemic Uncertainty Quantification in Organ Segmentation". To be presented at the [UNSURE workshop](https://unsuremiccai.github.io/) held in conjunction with MICCAI 2023. 

Models are defined via a `.yaml` configuration files. Example configuration files are provided in `cfgs/`.

## Model Training
To train a model, call `train.py` with the configuration file. For example:
```
python train.py -c cfgs/base.yaml
```
This will download the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset specified in the config file and train the model. 

If training a `swag` model, train a `base` model first then use `train_swag.py`. For example:
```
python train.py -c cfgs/base.yaml
```

If using an `ensemble` model, train each member separately then add them to a shared config file, such as in `cfgs/ensemble_base.yaml` for testing. 

## Model Testing
When training is complete, the `.yaml` file will be copied to the appropriate folder in the specified `work_dir` and the `best_model_path` will be added to it.

To test a `base` model, call `test_base.py` with the updated config file and the set to test (train, val, test, or all). For example:
```
python test_base.py -c experiments/Task09_Spleen/fold0/base_seed0/base.yaml -t test
```

To test any other model, call `test.py` with the config file, set to test, and number of smaples to use in uncertainty quantification. For example: 
```
python test.py -c experiments/Task09_Spleen/fold0/base_seed0/concrete_dropout.yaml -t test -n 4
```
If `-ttd True` is used, test time dropout will be used in prediciton.
If `-s True` is used, the predicted segmentations and uncertainty maps will be written to `.nrrd` files.

When `test_base.py` and `test.py` are called the results are printed and written to a `.json` file. Visualization of slices is also saved. 