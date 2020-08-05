# SemanticKITTI Experiments with PointNet++ as Backbone

This folder contains codes of experiments with PointNet++ as Backbone on SemanticKITTI dataset. Note that we follow the Single Scan task of the [SemanticKITTI Competition](https://competitions.codalab.org/competitions/20331).

## Training

To train the network, please first change to `tools/`

```
cd tools/
```

Under the directory of `tools/`, use following command to train a network from scratch:

```
python trainEval.py --data_root /path/to/dataset --output_dir /path/to/output
```

To train a network from a pretrained model:

```
python trainEval.py --data_root /path/to/dataset --output_dir /path/to/output --ckpt /path/to/pretrained
```

One may change the flag `--net` to use different frameworks (PNv2_SAP-1, PNv2_ASAP-1 or PNv2_ASAP-2) and other flags like `--batch_size`, `--lr` etc.

## Inference

To get the predictions of a trained model:

```
cd tools
python test.py --data_root /path/to/dataset --output_dir /path/to/output --ckpt /path/to/pretrained
```

To get the result on test split, you should submit the predictions to Single Scan task of the [SemanticKITTI Competition](https://competitions.codalab.org/competitions/20331). You may also check the validity of your submission using the [python script](https://github.com/PRBonn/semantic-kitti-api/blob/master/validate_submission.py) in [PRBonn](https://github.com/PRBonn)/**[semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)**.