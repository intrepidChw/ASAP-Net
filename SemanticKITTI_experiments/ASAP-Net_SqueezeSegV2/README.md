# SemanticKITTI Experiments with SqueezeSegV2 as Backbone

This folder contains codes of experiments with SqueezeSegV2 as Backbone on SemanticKITTI dataset. Note that we follow the Single Scan task of the [SemanticKITTI Competition](https://competitions.codalab.org/competitions/20331).

## Configuration files

Architecture configuration files are located at `train/tasks/semantic/config/arch`. Dataset configuration files are located at `train/tasks/semantic/config/labels`.

## Training

To train the network, please first change to `train/tasks/semantic`

```
cd train/tasks/semantic
```

Under the directory of `train/tasks/semantic`, use following command to train a network from scratch:

```
python seq_train.py -d /path/to/dataset -ac /config/arch/CHOICE.yaml -l /path/to/log
```

To train a network from a pretrained model:

```
python seq_train.py -d /path/to/dataset -ac /config/arch/CHOICE.yaml -l /path/to/log -p /path/to/pretrained
```

This will generate a tensorboard log, which can be visualized by running:

```
cd /path/to/log
tensorboard --logdir=. --port 5555
```

And acccessing [http://localhost:5555](http://localhost:5555/) in your browser.

## Inference

To get the predictions of a trained model:

```
cd train/tasks/semantic
python seq_infer.py -d /path/to/dataset/ -l /path/for/predictions -m /path/to/model
```

To get the result on test split, you should submit the predictions to Single Scan task of the [SemanticKITTI Competition](https://competitions.codalab.org/competitions/20331). You may also check the validity of your submission using the [python script](https://github.com/PRBonn/semantic-kitti-api/blob/master/validate_submission.py) in [PRBonn](https://github.com/PRBonn)/**[semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)**.