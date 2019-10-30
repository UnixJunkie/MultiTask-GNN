# Multitask Learning On Graph Neural Networks

In this repository we provide the code that is necessary to reproduce the results from the paper [Multitask Learning On Graph Neural Networks Applied To Molecular Property Predictions](https://arxiv.org/pdf/1910.13124.pdf). 


# Dependencies
<img width="40%" src="https://raw.githubusercontent.com/rusty1s/pytorch_geometric/master/docs/source/_static/img/pyg_logo_text.svg?sanitize=true" />


The list of libraries that you need to install to execute the code:
- torchvision=0.3.0
- networkx=2.3
- rdflib=4.2.2
- torch_spline_conv=1.1.0
- plyfile=0.7
- numpy=1.16.4
- six=1.12.0
- scipy=1.3.0
- torch_sparse=0.4.0
- googledrivedownloader=0.4
- torch_cluster=1.4.3
- torch=1.1.0
- matplotlib=3.1.0
- h5py=2.9.0
- pandas=0.24.2
- torch_scatter=1.3.1
- Pillow=6.1.0
- gdist=1.0.3
- scikit_learn=0.21.3

You can install all of those libraries through :

```
pip install -r requirements.txt
```

We have also included [pytorch geometric](https://github.com/rusty1s/pytorch_geometric) in the present repository. If you would like to install it instead, you need to have PyTorch 1.2.0 and install it through

```
pip install torch-geometric
```

The libraries [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-sparse](https://github.com/rusty1s/pytorch_sparse) and [torch-cluster](https://github.com/rusty1s/pytorch_cluster) need to be installed first. Please, have a look at the  [installation guide](https://github.com/rusty1s/pytorch_geometric) for further information.

## rdkit

[rdkit](https://github.com/rdkit/rdkit) cannot be installed through pip. Please, refer to their [installation page](https://github.com/rdkit/rdkit/blob/master/Docs/Book/Install.md) to install it on your system. 

# Usage

To test the algorithms, you need to change the parameters in *config.cfg* and run:

```
python3 main.py
```

This will run a k-fold cross-validation with a train/test split like explained in the paper.
The only possibilities for the parameter "model" are "GIN", "GGRNet" and "GAIN", which corresponds to the models
of the paper. 


# Citation

```
@article{mtlgnn2019,
    author  = {Fabio Capela and Vincent Nouchi and Ruud Van Deursen and Igor V. Tetko and Guillaume Godin},
    title   = {Multitask Learning On Graph Neural Networks Applied To Molecular Property Predictions},
    journal = {arXiv:1910.13124},
    year    = {2019}
}
```

# License
The code is freely available under a Clause-3 BSD license, as found in the LICENSE file
