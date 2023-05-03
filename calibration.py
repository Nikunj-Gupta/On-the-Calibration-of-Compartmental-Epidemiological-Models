import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt, os 
from scipy.integrate import odeint
from lmfit import minimize, Parameters
from sklearn.metrics import mean_absolute_error
from data_load import dataModel

'''
Reste a faire :
    - add mse out
    - sortie avec un json ?
    -comment the code
'''


#-----------Calibration-Model--------------#

class calibModel:

    def __init__(self):
        pass


    def model_deriv(self, y, t, N, params, cor_tab, nb_comp):
    
        dy = np.zeros(nb_comp)
        ind=0
        for i in range(nb_comp):
            for j in range(nb_comp):
                if cor_tab[i][j]==1: 
                    if ((i==0) and (j==1)):
                        dy[i]=dy[i]-(params[ind]*y[i]*y[1])/N
                        dy[j]=dy[j]+(params[ind]*y[i]*y[1])/N
                    else:
                        dy[i]=dy[i]-params[ind]*y[i]
                        dy[j]=dy[j]+params[ind]*y[i]
                    ind+=1
        return dy

    
    def solve(self, y0, t, N, params, cor_tab, nb_comp):

        res = odeint(self.model_deriv, y0, t, args=(N, params, cor_tab, nb_comp)) 
        result = res.T

        return result

    
    def err(self, params, t,  data, y0,  N, cor_tab, nb_comp, name_comp, name_params, nb_params, fit_tab):

        params_deriv = np.zeros(nb_params)
        for i in range(nb_params):
            params_deriv[i]=params[name_params[i]]

        result  = self.solve(y0, t, N, params_deriv, cor_tab, nb_comp)

        err = 0
        for i in range(nb_comp):
            if fit_tab[i]==1:
                err = pow((result[i,:] - data[name_comp[i]]),2)

        return err

    def disp(self,fitted_curve, data, name_comp, fit_tab):

        for i in range(len(name_comp)): 
            if name_comp[i] == 'Infected': 
                if fit_tab[i]==1:
                    plt.plot(fitted_curve[i,:], label=name_comp[i]+'_fitted')
                    plt.plot(data[name_comp[i]], '+', label=name_comp[i])
        plt.legend()
        plt.show()
    
    def disp(self, train_size, t, fit, data_nr, mae, methods, name, save_dir):	
        data_nr = data_nr['Infected']
        fig, ax = plt.subplots(figsize=(8.26, 8.26))
        # ax.set_ylim(0,data_nr.max()*1.1)
        ax.set_title('Infected')
        plt.axvline(x=train_size,color='gray',linestyle='--', label="End of train dataset")
        ax.scatter(t, data_nr, marker='+', color='black', label=f'Measures (method = {methods})')
        ax.plot(t, fit[:][1], 'g-', label=f'Simulation')
        ax.vlines(t, data_nr, fit[:][1], color='g', linestyle=':', label=f'MAE = {mae:.1f}')
        fig.legend(loc='upper center')
        # plt.show() 
        if not os.path.exists(save_dir+'/graphs/'):
            os.makedirs(save_dir+'/graphs/')
        plt.savefig(f'{save_dir}/graphs/graph_{name}_{methods}_{train_size}.png')
        plt.close(fig)

    def calib(self, file_json, set_gamma=False, params_out=False, graph_out=True, method='leastsq', max_nfev=1000, train_size=None, save_dir=None):

        d = dataModel()
        guess, N, t, t_train, data, y0, cor_tab, nb_comp, name_comp, name_params, fit_tab = d.load_config(file_json, train_size)

        nb_params = len(guess)

        #defining parameters
        params = Parameters() 
        for i in range(nb_params):
            params.add(name_params[i], value=guess[i], min=0, max=10, vary=True)
        params.add('N', value=N, vary=False)
         
        if set_gamma:
            params.add('Gamma', value=1, vary=False)

        #applying the fit
        out = minimize(self.err, params, method=method, args=(t_train, data.iloc[:train_size], y0, N, cor_tab, nb_comp, name_comp, name_params, nb_params, fit_tab),max_nfev=max_nfev)

        fitted_parameters = np.zeros(nb_params)
        for i in range(nb_params):
            fitted_parameters[i]=out.params[name_params[i]].value

        # fitted_parameters=(0.2,0.125)
        fitted_curve = self.solve(y0, t, N, fitted_parameters, cor_tab, nb_comp)

        # print(out.params)

        if params_out:
            d.out_results(out) 
        
        fitted_curve_dataframe = pd.DataFrame(columns = data.columns) 
        for i in range(nb_comp):
            fitted_curve_dataframe[name_comp[i]] = fitted_curve[i] 
        mae = mean_absolute_error(data['Infected'], fitted_curve_dataframe['Infected']) # MAE of simulated infected data points 

        if graph_out:
            # self.disp(fitted_curve, data, name_comp, fit_tab) 
            self.disp(train_size, t, fitted_curve, data, mae, method, name='_'.join(file_json.split('/')[-1].split('.')[0].split('_')[1:]), save_dir=save_dir) 

        return out, fitted_curve, fitted_parameters, mae 
