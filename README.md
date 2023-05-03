# On the Calibration of Compartmental Epidemiological Models 
Epidemiological compartmental models are useful for understanding infectious disease propagation and directing public health policy decisions. Calibration of these models is an important step in offering accurate forecasts of disease dynamics and the effectiveness of interventions. 

Full report: [Link]() coming soon! 

# Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Overview

In this study, we present an overview of calibrating strategies that can be employed, including several optimization methods and reinforcement learning (RL). We discuss the benefits and drawbacks of these methods and highlight relevant practical conclusions from our experiments. Optimization methods iteratively adjust the parameters of the model until the model output matches the available data, whereas RL uses trial and error to learn the optimal set of parameters by maximizing a reward signal. Finally, we  discuss how the calibration of parameters of epidemiological compartmental models is an emerging field that has the potential to improve the accuracy of disease modeling and public health decision-making. Further research is needed to validate the effectiveness and scalability of these approaches in different epidemiological contexts. 

# Installation

Create a new virtual environment `venv` for the repository and then activate it using: 

```
python -m venv venv 
source venv/bin/activate 
```

Then, simply run the following command to install all the required Python packages: 

```
pip install -r requirements.txt
```

The following packages are needed and will get installed: 

`pandas`, `numpy`, `lmfit`, `scipy`, `matplotlib`, `sklearn`, `gym` 

# Usage

## Simulating Epidemiological compartmental data for SIR, SIRD, or SIRVD* 
*as discussed in report. 

We use `scipy`'s `odeint` to simulate our data. 

Use the `model_gen.py` python script to generate / simulate data of your choice of configuration. 

The script takes the following four options: 

`--file_json`: Specifies the path to the input config file (required)

`--num_sim_days`: Specifies the desired number of days for simulation 

`--save_data`: True or 1 for saving the data at the path mentioned in configuration json file. 

`--plot`: True or 1 for displaying the simulated curves for visulaization 

Example to run the script with above options: 

```
python model_gen.py \
	--file_json configs/simple/config_SIR.json \
	--num_sim_days 175 \
	--plot 1 \
	--save_data 0 
```

## Adding noise 

To add noise to the previously simulated data, you can simply use the `--noise_level` option in `model_gen.py` python script to the desired noise level. 

Example for adding noise to data and saving it: 

```
python model_gen.py \
	--file_json configs/simple/config_SIR_noisy_level2.json \
	--num_sim_days 175 \
    --noise_level 0 \
	--plot 1 \
	--save_data 0 
```

## Considering Population subgroups 

## Simulating Epidemiological compartmental data for custom ODE 

First you will need to create a calibModel(), if you want to use the edge-adjacency matrix, or a calibModelOde(), if you want to use a custom ODE:

```python
from calibration import calibModel  
from calibration import calibModelEdo  
  
model = calibModel()
model2 = calibModelOde()
```

You will then have to call the calib function from those model. It will return
three values:  

* A minimizer result class (see [here](https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult) for more information)  
* A matrix containing the values of all comportment during the periode of
the data set producing the best fit.  
* An array made of the parameters producing the best fit.  

You can call the function in this way:  

```python
out, fitted_curve, fitted_parameters = model.calib("config_file_name.json")
```

If you want to use a custom Ode you will have to call it this way : 
python
```
out, fitted_curve, fitted_parameters = model2.calib("config_file_name.json", deriv)  
```
With **deriv** being the name of you ODE function defined as such : 

```python
def deriv(y, t, N, params):
   dy = np.zeros(3)
   S, I, R = y
   dy[0] = -params[0] * S * I / N
   dy[1] = params[0] * S * I / N - params[1] * I
   dy[2]= params[1] * I
   return dy
```

The calib function also have differents parameters : 

```python
calib(name_json, deriv (if using the Ode model), set_gamma=False, params_out=False, graph_out=True, method=’leastsq’, max_nfev=1000)
```

You can modify those parameters, given with their default values above:
* **set_gamma** take a Boolean. Sometimes you have access to newly infected data everyday. When set to True, this allows the user to manually set the recovery rate parameter to 1, basically telling the program to not expect continuity in the infected value over time, in order to still produce a fit. If false it will estimate this parameter.
* **params_out** take a Boolean. If true, it will output the parameters in a .txt file located in the calibration/out folder.
* **graph_out** take a Boolean. If true, it will display a graph of the fit, allowing you to monitor its coherence.
* **method** take a string. It allows you to choose which method to use when calibrating the model. You can choose them from the methods.txt file
* **max_nfev** take an integer. It allows you to set a maximum number of iterations for the calibration 

A sample config json file: 
```
{
    "guess" : [],
    "nb_comp" : 0,
    "name_comp" : [],
    "cor_tab" : [[],[]],
    "name_params" : [],
    "fit_tab" : [],

    "N" : 0,
    "y0" : [],

    "name_fic" : ""
}
```
## Calibration: Running optimization methods 

## Calibration: Running our Reinforcement Learning Approach 

## Generating plots: Superimposing predicted over real data to visualize the error 

# Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them to your branch
4. Push your changes to your fork
5. Create a pull request from your branch to the main branch of this repository

Please make sure to follow our code style and formatting guidelines, and to write clear and concise commit messages. We also encourage you to add unit tests and documentation for any new features or changes.

By contributing to this project, you agree to license your contributions under the same license as this project (see [License](#license) section for details).

If you have any questions or feedback, please don't hesitate to open an issue or reach out to us directly. We appreciate your contributions and look forward to working with you! 