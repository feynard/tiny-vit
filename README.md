This is a simple implementation of ViT model. Example includes training of classifier on `CIFAR-10`. Requirements are as usual:
- ``pytorch``
- ``pyyaml``

and relatively fresh ``Python``.

To download and preprocess the data into a given folder run
```
python build_dataset.py ./cifar-10
```

To train the model run
```
python train.py config.yaml
```
or alternatively look at the provided ``train.ipynb``.

*Note: the default config is intentionally defines a tiny model*
