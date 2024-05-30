# Early mutational signatures and transmissibility of SARS-CoV-2 Gamma and Lambda variants in Chile

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the code repository to reproduce the figures and analysis from our "Mutational signatures and transmissibility of SARS-CoV-2 Gamma and Lambda variants" publication. 

## Reproducing figures

The figures from our publication can be reproduced using the `notebooks/create_plots.ipynb` notebook. Before you can do that you have to run the analysis by having a look into the `notebooks/Model.ipynb` or the `src/run_model.py` script. 

We advice you start with the `notebooks/Model.ipynb` notebook and than take a look into the `src/run_model.py` once necessary.

We also supply a small trace in the data folder which can be loaded if you don't want to run the time consuming sampling. 


## Notes

The herein presented code highly relies on our toolbox for [Bayesian python toolbox for inference and forecast of the spread of the Coronavirus](https://github.com/Priesemann-Group/covid19_inference/tree/v0.3.1). If you want to run our code make sure to install the toolbox or to initialize the github submodule.

```bash
git submodule update --init
```
