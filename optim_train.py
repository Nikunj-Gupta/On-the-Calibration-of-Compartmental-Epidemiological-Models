from calibration import calibModel 
from model_gen import model_deriv, data_gen 
from scipy.integrate import odeint 
import numpy as np, pandas as pd, json, argparse 
from sklearn.metrics import mean_squared_error 
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='SIR') 
parser.add_argument('--gen_data', type=int, default=1) 
parser.add_argument('--file_json', type=str, default="configs/simple/config_SIR.json") 
parser.add_argument('--method', type=str, default=None) 
parser.add_argument('--save_dir', type=str, default="results/") 
parser.add_argument('--start', type=int, default=25) 
parser.add_argument('--end', type=int, default=85) 
parser.add_argument('--step', type=int, default=10) 
parser.add_argument('--plot', type=int, default=0) 

args = parser.parse_args()


with open(args.file_json, "r") as read_file:
    d = json.load(read_file) 
name_params = d["name_params"]
true_params = d["true_params"] 

results = {
    'Model': [], 
    'Training_days': [], 
    'Method': [], 
    'Mae': [], 
    'mse': []
}
for name in name_params: results[name] = []  


model = calibModel()

if args.method: 
     methods = [args.method]
else: 
    methods=[
            "leastsq", 'differential_evolution', 'brute', 'basinhopping', 
            'ampgo', 'nelder', 'lbfgsb', 'powell', 'cg', 
            'bfgs', 'trust-constr', 'tnc', 'slsqp', 
            'shgo', 'dual_annealing'
        ] 

for method in methods: 
        for train_size in range(args.start, args.end, args.step): 
            out, fitted_curve, fitted_parameters, mae = model.calib( 
                                                            args.file_json, 
                                                            set_gamma=False, 
                                                            params_out=False, 
                                                            graph_out=args.plot, 
                                                            method=method, 
                                                            max_nfev=1000, 
                                                            train_size=train_size, 
                                                            save_dir=args.save_dir
                                                        ) 
            mse = mean_squared_error(true_params, fitted_parameters) 

            results['Model'].append(args.model_name) 
            results['Training_days'].append(train_size) 
            results['Method'].append(method) 
            for i, name in enumerate(name_params): results[name].append(fitted_parameters[i]) 
            results['Mae'].append(mae) 
            results['mse'].append(mse) 

results = pd.DataFrame(results) 
print(results)

if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
save_path = (args.save_dir +"/results_"+ '_'.join(args.file_json.split('/')[-1].split('_')[1:])).split('.')[0] + '.csv' 
results.to_csv(save_path, index=False) 
