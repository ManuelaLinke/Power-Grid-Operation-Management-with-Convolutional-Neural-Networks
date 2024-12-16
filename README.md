# Power Grid Operation in Distribution Grids with Convolutional Neural Networks

[![Python version](https://img.shields.io/badge/python-3.10.*-violet.svg)](https://img.shields.io/badge/python-3.10.*-violet)
[![Funding](https://img.shields.io/badge/Project-AI4Grids-%23fcba03?link=https%3A%2F%2Fwww.htwg-konstanz.de%2Fhochschule%2Fprojekte%2Fai4grids%2Fueber-ai4grids%2F.svg)](https://www.htwg-konstanz.de/hochschule/projekte/ai4grids/ueber-ai4grids/)
[![Linkedin](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/manuela-linke/)

This repository contains the Python code to reproduce the results of our paper published by Smart Energy Journal (doi:10.1016/j.segy.2024.100169).

This project is used for the development and investigation of a grid optimization tool based on convolutional neural networks (CNNs) with the aim of avoiding supply bottlenecks through intelligent use of the existing grid infrastructure and thus minimizing the need for grid expansion measures.

This project is developed and maintained by [Manuela Linke](https://www.researchgate.net/profile/Manuela-Linke) at the [HTWG Konstanz](https://www.htwg-konstanz.de/). Parts of the code were developed by [Tobias Meßmer](https://www.researchgate.net/profile/Tobias-Messmer-2) and [Gabriel Micard](https://www.researchgate.net/profile/Gabriel-Micard). 

![graphical_abtract](/doc/img/Graphical_Abstract_V4.png)

## Highlights

-   Grid operation based on convolutional neural networks with a maximum accuracy of 99.06 %
-   Application illustrated on real world scenario with virtual grid
-   Two approaches investigated for the implementation of input data
-   Paving the way for further integration of renewable energy sources as well as heat pumps and electrical cars into the existing grid without grid expansion


## Dependencies

This project is written and tested to be compatible with Python 3.10.

It leans heavily on the following Python packages:

-   [pandapower](http://pandapower.org/) for power system modeling, analysis and optimization. 
-   [numpy](http://www.numpy.org/) for calculations, such as linear algebra and matrix calculations
-   [matplotlib](https://matplotlib.org/) for plotting
-   [tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the CNN model 
-   [jupyterlab](https://jupyter.org/) as interactive development environment

  
## Installation

Create Environment:

```conda create -n cnn_regler python=3.10.*```

```pip install -r requirements.txt```

For using the script to generate the training data: 

```python 2_Generate_Training_Data.py . data --max_workers=30```

Note: adjust maximum number of workers to you number of kernels.


## Screenshots

| The virtual grid used in this project | Flowchart of the training data generation algorithm |
|:-------------------------------------:|:---------------------------------------------------:|
| ![virtual grid](/doc/img/Cossmic_grid_EN-1.png) | ![flowchart](/doc/img/Dataset_generation.png) |


## Generated Datasets

The Results presented in the associated paper are based oon the generated training datasets with the timestamp:
-   2024-03-22_13-06-27 (smaller dataset to comapre the approaches)
-   2024-03-25_16-56-21 (bigger dataset used for extended training of the physical approach)
  

## For using your own data
For using these scripts you need to take care, that 
- the order of nodes in you pandapower grid is the same as the order of loads in the grid
- If you have nodes that are not connected to a load you need to implement an exeption
- generations are considered as negative loads
- loads has to have an additional row for p_max

Follow the numeration of the scripts to execute the code.


## Contributing and Support

We warmly invite anyone interested in contributing to this project. If you have any ideas, suggestions, or encounter any problems, please feel free to contact us.


## Citing 

If you use this project for your research,  we kindly request that you cite the following paper:

-   M. Linke, T. Messmer, G. Micard, G. Schubert, [Power Grid Operation in Distribution Grids with Convolutional Neural Networks](), 2024, [Smart Energy](https://www.sciencedirect.com/journal/smart-energy), 10.1016/j.segy.2024.100169


Please use the following BibTeX:

@article{LINKE2025100169,
  title = {Power grid operation in distribution grids with convolutional neural networks},
  journal = {Smart Energy},
  volume = {17},
  pages = {100169},
  year = {2025},
  issn = {2666-9552},
  doi = {https://doi.org/10.1016/j.segy.2024.100169},
  url = {https://www.sciencedirect.com/science/article/pii/S266695522400039X},
  author = {Manuela Linke and Tobias Meßmer and Gabriel Micard and Gunnar Schubert},
  keywords = {Power grid operation, Convolutional neural network, Artificial intelligence, Smart grids, Resilient energy system, Sector coupling},
  abstract = {The efficient and reliable operation of power grids is of great importance for ensuring a stable and uninterrupted supply of electricity. Traditional grid operation techniques have faced challenges due to the increasing integration of renewable energy sources and fluctuating demand patterns caused by the electrification of the heat and mobility sector. This paper presents a novel application of convolutional neural networks in grid operation, utilising their capabilities to recognise fault patterns and finding solutions. Different input data arrangements were investigated to reflect the relationships between neighbouring nodes as imposed by the grid topology. As disturbances we consider voltage deviations exceeding 3% of the nominal voltage or transformer and line overloads. To counteract, we use tab position changes of the transformer stations as well as remote controllable switches installed in the grid. The algorithms are trained and tested on a virtual grid based on real measurement data. Our models show excellent results with test accuracy of up to 99.06% in detecting disturbances in the grid and suggest a suitable solution without performing time-consuming load flow calculations. The proposed approach holds significant potential to address the challenges associated with modern grid operation, paving the way for more efficient and sustainable energy systems.}
}


## Licence

This project is licensed under [Apache License 2.0](LICENSE).

## Acknowledgements

This research was funded by the [Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (BMUV)](https://www.bmuv.de/) based on a resolution of the German Bundestag as part of the Reasearch Project [AI4Grids](https://www.htwg-konstanz.de/hochschule/projekte/ai4grids/ueber-ai4grids/).

The authors would like to thank [Marcel Arpogaus](https://github.com/MArpogaus) for implementation of the parallelisation of the training process. His contributions have significantly reduced the computational time for the training data generation. Furthermore, the authors thank Jan Weccard for his assistance with the first implementation of the physical approach.
