<!--
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/ita-ml%2Finstance-hardness/binder?filepath=notebooks%2F)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://en.wikipedia.org/wiki/MIT_License)
-->

# PyHard

_Instance Hardness Python package_

<!--![picture](docs/img/circle-fs.png)-->

## Getting Started

PyHard employes a methodology known as [_Instance Space Analysis_](https://github.com/andremun/InstanceSpace) (ISA) to analyse performance at the instance level rather than at dataset level. The result is an alternative for visualizing algorithm performance for each instance within a dataset, by relating predictive performance to estimated instance hardness measures extracted from the data. This analysis reveals regions of strengths and weaknesses of predictors (aka _footprints_), and highlights individual instances within a  dataset that warrant further investigation, either due to their unique properties or potential data quality issues.


### Installation
Although the original ISA toolkit has been written in Matlab, we provide a lighter version in Python, with less tools, but enough for the instance hardness analysis purposes. You may find the implementation in the separate package [PyISpace](https://gitlab.com/ita-ml/pyispace). Notwithstanding, the choice of the ISA engine is left up to the user, which can be set in the configuration file. Below, we present the standard installation, and also the the additional steps to configure the Matlab engine (optional).

_Standard installation_

```
pip install pyhard
```


Alternatively,
```
git clone https://gitlab.com/ita-ml/pyhard.git
cd pyhard
pip install -e .
```


#### Additional steps for Matlab engine (optional)

Python 3.7 is required. Matlab is also required in order to run the ISA source code. As far as we know, only recent versions of Matlab offer an engine for Python 3. Namely, we only tested from version R2019b on.

<!--Alternatively, take a look at [_Graphene_](https://gitlab.com/ita-ml/graphene), the Instance Hardness Analytics Tool. Matlab, and even Python, are not required in this case!-->

1. __Install Matlab engine for Python__  
Refer to this [link](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), which contains detailed instructions.

2. __Clone the ISA repository__  
You may find it [here](https://github.com/andremun/InstanceSpace).

3. __Change config file__  
In `config.yaml`, set the fields `isa_engine: matlab` and `matildadir: path/to/isa_folder` (cloned in step 2)


### Usage

First, make sure that the configuration files are placed within the current directory and with the desired settings. Otherwise, see this [section](#configuration) for more details.

Then, in the command line, run:

```
pyhard
```

By default, the following steps shall be taken:

1. Calculate the _hardness measures_;

2. Evaluate classification performance at instance level for each algorithm;

3. Select the most relevant hardness measures with respect to the instance classification error;

4. Join the outputs of steps 1, 2 and 3 to build the _metadata_ file (`metadata.csv`);

5. Run __ISA__ (_Instance Space Analysis_), which generates the _Instance Space_ (IS) representation and the _footprint_ areas;

6. To explore the results from step 5, launch the visualization dashboard:  
``pyhard --app``


One can choose which steps should be disabled or not

* `--no-meta`: does not attempt to build the metadata file

* `--no-isa`: does not run the Instance Space Analysis


To see all command line options, run `pyhard --help` to display help.


### Input file

Please follow the guidelines below:

* Only `csv` files are accepted

* The dataset should be in the format `(n_instances, n_features)`

* **Do not** include any index column. Instances will be indexed in order, starting from **1**

* **The last column** must contain the classes of the instances

* Categorical features should be handled previously


### Configuration

Inside the folder where the command `pyhard` is run, make sure that the files `config.yaml` and `options.json` are present. They contain configurations for PyHard and ISA respectivel. One may generate them locally with command `pyhard -F`.

The file `config.yaml` is used to configurate steps 1-4. Through it, options for file paths, measures, classifiers, feature selection and hyper-parameter optimization can be set. Inside the file, more instructions may be found.

A configuration file in another location can be specified in the command line: 
`pyhard -c path/to/new_config.yaml`


### Visualization

#### Demo

<!--![picture](docs/img/demo.png)-->

The demo visualization app can display any dataset located within `pyhard/data/`. Each folder within this directory (whose name is the problem name) should contain those three files:

- `data.csv`: the dataset itself;

- `metadata.csv`: the metadata with measures and algorithm performances (`feature_` and `algo_` columns);

- `coordinates.csv`: the instance space coordinates.

The showed data can be chosen through the app interface. To run it use the command:

```
pyhard --demo
```

New problems may be added as a new folder in `data/`. Multidimensional data will be reduced with the chosen dimensionality reduction method.

#### App

<!--![picture](docs/img/animation.gif)-->

Through command line it is possible to launch the app for visualization of 2D-datasets along with their respective instance space. The graphics are linked, and options for color and displayed hover are available. In order to run the app, use the command:

```
pyhard --app
```

It should open the browser automatically and display the data.


## References

_Base_

1. Michael R. Smith, Tony Martinez, and Christophe Giraud-Carrier. 2014. __An instance level analysis of data complexity__. Mach. Learn. 95, 2 (May 2014), 225–256.

2. Ana C. Lorena, Luís P. F. Garcia, Jens Lehmann, Marcilio C. P. Souto, and Tin Kam Ho. 2019. __How Complex Is Your Classification Problem? A Survey on Measuring Classification Complexity__. ACM Comput. Surv. 52, 5, Article 107 (October 2019), 34 pages.

3. Mario A. Muñoz, Laura Villanova, Davaatseren Baatar, and Kate Smith-Miles. 2018. __Instance spaces for machine learning classification__. Mach. Learn. 107, 1 (January   2018), 109–147.

_Feature selection_

4. Luiz H. Lorena, André C. Carvalho, and Ana C. Lorena. 2015. __Filter Feature Selection for One-Class Classification__. Journal of Intelligent and Robotic Systems 80, 1 (October   2015), 227–243.

5. Artur J. Ferreira and MáRio A. T. Figueiredo. 2012. __Efficient feature selection filters for high-dimensional data__. Pattern Recognition Letters 33, 13 (October, 2012), 1794–1804.

6. Jundong Li, Kewei Cheng, Suhang Wang, Fred Morstatter, Robert P. Trevino, Jiliang Tang, and Huan Liu. 2017. __Feature Selection: A Data Perspective__. ACM Comput. Surv. 50, 6, Article 94 (January 2018), 45 pages.

7. Shuyang Gao, Greg Ver Steeg, and Aram Galstyan. __Efficient Estimation of Mutual Information for Strongly Dependent Variables__. Available in http://arxiv.org/abs/1411.2003. AISTATS, 2015.

_Hyper parameter optimization_

8. James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. __Algorithms for hyper-parameter optimization__. In Proceedings of the 24th International Conference on Neural Information Processing Systems (NIPS’11). Curran Associates Inc., Red Hook, NY, USA, 2546–2554.

9. Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. 2012. __Practical Bayesian optimization of machine learning algorithms__. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 2 (NIPS’12). Curran Associates Inc., Red Hook, NY, USA, 2951–2959.
  
10. J. Bergstra, D. Yamins, and D. D. Cox. 2013. __Making a science of model search: hyperparameter optimization in hundreds of dimensions for vision architectures__. In Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28 (ICML’13). JMLR.org, I–115–I–123.
  
