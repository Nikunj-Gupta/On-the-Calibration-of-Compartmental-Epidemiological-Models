# On the Calibration of Compartmental Epidemiological Models 
Epidemiological compartmental models are useful for understanding infectious disease propagation and directing public health policy decisions. Calibration of these models is an important step in offering accurate forecasts of disease dynamics and the effectiveness of interventions. 

Full report: [Link]() coming soon! 

# Table of Contents

- [Overview](#overview)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

# Overview

In this study, we present an overview of calibrating strategies that can be employed, including several optimization methods and reinforcement learning (RL). We discuss the benefits and drawbacks of these methods and highlight relevant practical conclusions from our experiments. Optimization methods iteratively adjust the parameters of the model until the model output matches the available data, whereas RL uses trial and error to learn the optimal set of parameters by maximizing a reward signal. Finally, we  discuss how the calibration of parameters of epidemiological compartmental models is an emerging field that has the potential to improve the accuracy of disease modeling and public health decision-making. Further research is needed to validate the effectiveness and scalability of these approaches in different epidemiological contexts. 

# Background 

One of the simplest compartmental models is the SIR model (Suspected, Infected, Recovered) and it can be described as shown in the figure below: 

![SIR model](readme_assets/sir.png) 

Thus, we obtain the following equations with the parameters $\beta$ and $\gamma$ as transition rates: 

$$
\begin{cases}
    \frac{dS}{dt} = -\frac{\beta I S}{N} \\ \\
    \frac{dI}{dt} = \frac{\beta I S}{N} - \gamma I \\ \\
    \frac{dR}{dt} = \gamma I 
\end{cases}
$$



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

`pandas`, `numpy`, `lmfit`, `scipy`, `matplotlib`, `scikit-learn`, `gym` 


# Usage

## Simulating Epidemiological compartmental data for SIR, SIRD, or SIRVD* 
*as discussed in report. 

We use `scipy`'s `odeint` to simulate our data. 

Use the `model_gen.py` python script to generate / simulate data of your choice of configuration. 

The script takes the following four options: 

`--file_json`: Specifies the path to the input config file (required)

`--num_sim_days`: Specifies the desired number of days for simulation 

`--save_data`: True or 1 for saving the data at the path mentioned in configuration json file 

`--plot`: True or 1 for displaying the simulated curves for visulaization 

Example to run the script with above options: 

```
python model_gen.py \
	--file_json configs/simple/config_SIR.json \
	--num_sim_days 175 \
	--plot 1 \
	--save_data 0 
```


### Generalization of epidemiological models 

Let N be the number of compartments in our epidemiological model. We epresent a huge variety of model by using an edge-adjacency matrix. Let A be a NxN size edge-adjacency matrix, we define it as the following: 

$$
\mathrm{[A]}_{ij} = \begin{cases}
    1 & \text{if } \text{i and j are adjacent} \\ 
    0 & \text{otherwise}
\end{cases}
$$


For example, for simple SIR, the edge-adjacency matrix will look like this: 

$$
\begin{pmatrix}
    0 & 1 & 0\\
    0 & 0 & 1\\
    0 & 0 & 0
\end{pmatrix}
$$

This representation allows for a quick and fairly easy way to implement simple custom models. 

### The configuration file 


In order to properly set up the calibration, you will need to modify the configuration `json` file in the `configs` folder. Here is a general template: 


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


The configuration file needs the following information: 

* `guess`: is a matrix made of some estimation of the parameters to fit. 

* `nb_comp`: represents the number of compartments within your model. 

* `name_comp`: is a matrix made of the name of the compartments. Those names has to be the same as the header in the csv file. 

* `cor_tab`: is the edge-adjacency matrix discussed earlier. It is needed only if not using a custom ODE. (Details for custom ODE are discussed later in this document). 

* `fittab`: is a matrix with 1 in the ith row if you want to fit with the ith compartment. It is useful if you want to fit a complex model without having all the data available.

* `N`: represents the total population from the dataset 

* `y0`: represents the initial vector of value at day 0, we decided to not extract it from the data since a high variance in the data may lead to a flawed initial vector. 

* `name_fic`: is the name of the .csv file containing the data. It has to be stored in the calibration/data folder. 

## Adding noise 

To add noise to data, you can simply use the `--noise_level` option in `model_gen.py` python script to the desired noise level. 

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

We used [Epipolicy](https://epipolicy.github.io/) for generating data for simulated epidemics considering population subgroups. Its documentation ([here](https://epipolicy.github.io/docs_and_tutorials)) has a detailed explanation of how to use it. The exact contact matrices that we used for our experiments on age-based subgroups can be found in our [full report]() (chapter 4). 

The configuration json file used for Epipolicy can be found here as an example and for reference: `configs/epipolicy/SIR.json`. 

Corresponding generated sample data (3 age-based population subgroups) can be found here: `sample_data/epipolicy/simple/*` 

We also provide the sample data with added noise for reference: `sample_data/epipolicy/noisy/*` 


## Calibration: Running optimization methods 

Use the `optim_train.py` python script to train any / all optimization methods on your data.  

The script takes the following four options: 


`--model_name`: Specifies the name of the epidemic model, eg, SIR. 

`--gen_data`: True if you want to generate a csv of the results. The results will have the following columns: `Model`, `Training_days`, `Method`, `Mae`, `mse`, `<followed by one column for predictions for each epidemic model parameter>`. 

`--file_json`: Specifies the path to the input config file (required)

`--method`: Specifies the optimization method you want to use. List of all options: `leastsq`, `differential_evolution`, `brute`, `basinhopping`, `ampgo`, `nelder`, `lbfgsb`, `powell`, `cg`, `bfgs`, `tnc`, `trust-constr`, `slsqp`, `shgo`, `dual_annealing`, `least_squares`. If not specified, the scipt will run for all methods. 

`--save_dir`: Specifies path to directory to save results (csv and plots if option is selected)

`--start`, `--end`, `--step`: Together, these options specify the number of days available for training. Using these options, you can run the optimization methods for numerous amounts of days --- from 'start' to 'end' with a step size of 'step'. Example: To generate all results for all optimization methods for the following amounts of days available for training: [5, 10, 15, 20, 25] can be done using `--start 5 --end 25 --step 5` 

`--plot`: True if you want plot and save resulting graphs. The graphs plot the predicted curve superimposed over the real values and can be helpful for visualizing the errors in the predictions. 

Example to run the script with above options: 

```
python optim_train.py \
    --model_name SIR \
    --gen_data 1 \
    --file_json configs/simple/config_SIR.json \
    --numethodm_sim_days leastsq \
    --save_dir results/ \
    --start 5 \
    --end 25 \
    --step 5 \
    --plot 1 
```


## Calibration: Running our Reinforcement Learning Approach 

We use Proximal Policy Optimization in our RL approach. It is implemented in `ppo.py` python script. You are welocme to use any other RL algorithm. 

Example to run our RL script: 



```
python rl/rl.py
```


## Simulating Epidemiological compartmental data for custom ODE 

First you will need to create `calibModelOde()`, if you want to use a custom ODE:

```python
from calibration import calibModelEdo  
  
model = calibModelOde()
```

You will then have to call the `calib` function. It will return three values:  

* A minimizer result class (see [here](https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult) for more information) 

* A matrix containing the values of all comportment during the periode of the data set producing the best fit.  

* An array made of the parameters producing the best fit.  

You can call the function in the following way:  

```python
out, fitted_curve, fitted_parameters = model.calib("config_file_name.json", deriv)  
```

Here, **deriv** is your custom ODE function which needs to be defined as follows: 

```python
def deriv(y, t, N, params):
   dy = np.zeros(3)
   S, I, R = y
   dy[0] = -params[0] * S * I / N
   dy[1] = params[0] * S * I / N - params[1] * I
   dy[2]= params[1] * I
   return dy
```

Other options available in calib are: 

* `set_gamma` takes a Boolean. Sometimes you have access to newly infected data everyday. When set to True, this allows the user to manually set the recovery rate parameter to 1, basically telling the program to not expect continuity in the infected value over time, in order to still produce a fit. If false it will estimate this parameter.

* `params_out` takes a Boolean. If true, it will output the parameters in a `.txt` file located in the specified folder (`calibration/out` set for now).

* `graph_out` takes a Boolean. If true, it will display a graph of the fit, allowing you to monitor its coherence.

* `method` takes a string. It allows you to choose which method to use when calibrating the model. List of all options: `leastsq`, `differential_evolution`, `brute`, `basinhopping`, `ampgo`, `nelder`, `lbfgsb`, `powell`, `cg`, `bfgs`, `tnc`, `trust-constr`, `least_squares`, `slsqp`, `shgo`, `dual_annealing`. 

* `max_nfev` takes an integer. It allows you to set a maximum number of iterations for the calibration 

Example: 
```python
calib(name_json, deriv, set_gamma=False, params_out=False, graph_out=True, method=`leastsq`, max_nfev=1000)
```


# Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them to your branch
4. Push your changes to your fork
5. Create a pull request from your branch to the main branch of this repository

Please make sure to follow our code style and formatting guidelines, and to write clear and concise commit messages. We also encourage you to add unit tests and documentation for any new features or changes.

If you have any questions or feedback, please don't hesitate to open an issue or reach out to us directly (email: [ng2531@nyu.edu](mailto:ng2531@nyu.edu)). We appreciate your contributions and look forward to working with you! 

