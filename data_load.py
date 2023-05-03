import os
import pandas as pd
import numpy as np
import json

class dataModel:

    @staticmethod
    def load_config(file_json, train_size, upper_lim_data=1500):

        with open(file_json, "r") as read_file:
            d = json.load(read_file)

        guess = d["guess"]
        nb_comp = d["nb_comp"]
        name_comp = d["name_comp"]
        cor_tab = d["cor_tab"]
        name_params = d["name_params"]
        fit_tab = d["fit_tab"]

        N = d["N"]
        y0 = d["y0"]

        file_fic = d["file_fic"]
        data = pd.read_csv(file_fic) 
        if upper_lim_data: 
            data = data.iloc[:upper_lim_data]

        n = len(data)

        t = np.linspace(0, n, n)
        t_train = np.linspace(0, train_size, train_size)

        return guess, N, t, t_train, data, y0, cor_tab, nb_comp, name_comp, name_params, fit_tab  

    @staticmethod
    def out_results(out, out_path):
        
        with open(out_path+"/out.txt", "w") as outfile:
            outfile.write(str(out.params))
