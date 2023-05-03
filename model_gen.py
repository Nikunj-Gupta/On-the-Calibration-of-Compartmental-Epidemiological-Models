#FIRST PARAMETER MUST BE FROM S TO I

import numpy as np, os, json, argparse 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv

NUMBER_OF_DAYS = 175 
def init(file_json=None, num_sim_days=175): #initialise the data 
    with open(file_json, "r") as read_file:
        d = json.load(read_file)

    size = d["nb_comp"]
    name_tab = d["name_comp"] 
    cor_tab = np.array(d["cor_tab"]) 
    params = d["true_params"]

    N = d["N"]
    y0 = d["y0"]

    n = num_sim_days 
    t = np.linspace(0, n, n)
    
    
    file_fic = d["file_fic"]
    name_params = d["name_params"]
    fit_tab = d["fit_tab"]


    return name_tab, cor_tab, size, params, y0, N, n, t, file_fic 
    

def model_deriv(y , t, N, params, cor_tab, size):
    
    #print(y)
    dy = np.zeros(size)
    ind=0
    
    for i in range(size):
        for j in range(size):
            if cor_tab[i][j]==1: 
                if ((i==0) and (j==1)):
                    dy[i]=dy[i]-params[ind]*y[i]*y[1]/N
                    dy[j]=dy[j]+params[ind]*y[i]*y[1]/N
                else:
                    dy[i]=dy[i]-params[ind]*y[i]
                    dy[j]=dy[j]+params[ind]*y[i]
                ind+=1
    return dy


def data_gen(name_tab, size, result, n, file_fic):

    header = name_tab
    ficnamecsv=file_fic


    if not os.path.exists(os.path.dirname(ficnamecsv)):
        os.makedirs(os.path.dirname(ficnamecsv)) 

    with open(ficnamecsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
       
        for i in range(n):
            
            data = np.zeros(size)

            for j in range (size):
                data[j] = int(result[j][i])

            writer.writerow(data)


#------------------------Main-----------------------#

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--file_json', type=str, default="testML_subgroups/config/config_SIR_example.json", help='A required integer positional argument') 
    parser.add_argument('--num_sim_days', type=int, default=175, help='num_sim_days') 
    parser.add_argument('--save_data', type=int, default=1, help='save_data') 
    parser.add_argument('--noise_level', type=int, default=0, help='noise_level') 
    parser.add_argument('--plot', type=int, default=1, help='plot or not') 
    args = parser.parse_args()

    name_tab, cor_tab, size, params, y0 , N, n, t, file_fic = init(file_json=args.file_json, num_sim_days=args.num_sim_days) 

    res = odeint(model_deriv, y0, t, args=(N, params, cor_tab, size)) 
    result = res.T 

    noise = np.random.normal(scale=(args.noise_level*result.std())/100, size=result.shape) 
    print(result.std())

    result = result + noise
    
    if args.save_data: 
        data_gen(name_tab, size, result, n, file_fic)

    if args.plot: 
        for i in range(size):
            # if name_tab[i] == "Infected": 
                plt.plot(t, result[i,:], alpha=0.5, lw=2, label=name_tab[i])
        plt.legend()
        plt.show()