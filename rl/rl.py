import numpy as np, pandas as pd, argparse, gym, json, math, os 
from gym import spaces 

from scipy.integrate import odeint 
from sklearn.metrics import mean_absolute_error 

from tqdm import tqdm 
import matplotlib.pyplot as plt  
from ppo import PPO

from itertools import count
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter 


parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--model_complexity', type=str, default="SIRD") 
parser.add_argument('--noise_level', type=int, default=0) 
parser.add_argument('--sub_group', type=int, default=1) 
parser.add_argument('--amount_of_data', type=int, default=50)  
parser.add_argument('--data_file', type=str, default="data/data_SIRD_grp1.csv") 
parser.add_argument('--config_file', type=str, default="configs/config_SIRD_grp1.json") 
parser.add_argument('--eval_file', type=str, default="data/rl_eval/eval_sird_debug.npy") 


parser.add_argument('--max_episodes', type=int, default=100) 
parser.add_argument('--max_ep_len', type=int, default=30) 
parser.add_argument('--eval_freq', type=int, default=10) 
parser.add_argument('--std', type=int, default=1) 

parser.add_argument('--logdir', type=str, default="rl_logs/") 
parser.add_argument('--update_timestep', type=int, default=30) 
parser.add_argument('--save_model_freq', type=int, default=30) 


args = parser.parse_args()


TOTAL_TIMESTEPS = args.max_episodes # 75K 
MAX_EP_LENGTH = args.max_ep_len
EVAL_FREQ = args.eval_freq 
STD = args.std  

MODEL = args.model_complexity 
NOISE = args.noise_level 
SUB_GROUP = args.sub_group 
AMOUNT_OF_DATA = args.amount_of_data 
LOG_NAME = "--".join([
    MODEL, 
    "noise_"+str(NOISE), 
    "subgroup_"+str(SUB_GROUP), 
    "amount_of_data_"+str(AMOUNT_OF_DATA) 
])

# TOTAL_TIMESTEPS = 75_000  
MAE_MEAN = 22695090589.648327
MAE_STD = 914953560823.3027
OBS_LOW = 0. 
OBS_HIGH = 1. 

# Config 
with open(args.config_file) as f: 
    config = json.load(f) 

PARAMS = config["name_params"] # ["beta", "gamma"] 
NUM_PARAMS = len(PARAMS) 
n = 175 
y0 = config["y0"] # [299999999, 1, 0] 
t = np.linspace(0, n, n) 
N = config["N"] # 3e8 
cor_tab = config["cor_tab"] # [[0,1,0], [0,0,1], [0,0,0]] 
nb_comp = config["nb_comp"] # 3 
true_params = config["true_params"] # [0.4, 0.075] 
name_comp = config["name_comp"] # ["Suspected", "Infected", "Recovered"] 


hyperparams = {
    "max_episodes":TOTAL_TIMESTEPS,
    "max_cycles":MAX_EP_LENGTH,
    "update_timestep": args.update_timestep, 
    "logs_dir": args.logdir,
    "save_model_freq": args.save_model_freq, 
    "action_std_decay_rate": 0.05, 
    "action_std_decay_freq": int(2.5e5), 
    "min_action_std": 0.1, 
    "lr_actor": 0.0003, 
    "lr_critic": 0.001, 
    "gamma": 0.99, 
    "K_epochs": 8, 
    "eps_clip": 0.2, 
    "has_continuous_action_space": False, 
    "action_std": 0.5,
    "action_std_init": 0.6 
}

"""
Actions 

Increase/Decrease by 0.1, 0.01, 0.001 for both parameters + No change 
"""
actions = ["no change"] 
for k in PARAMS: 
    for i in ["increase", "decrease"]: 
        for j in [0.1, 0.01, 0.001]: 
            actions.append(tuple((k, i, j)))
ACT_DIM = len(actions) 

print("Actions: ") 
[print(action) for action in actions] 
print("Number of actions: ", ACT_DIM) 

"""
Compartmental Epidemic data 
"""

data = pd.read_csv(args.data_file) 
# print(data) 

def model_deriv(y, t, N, params, cor_tab, nb_comp):
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

"""
actions 
"""
def calc_params_from_actions(act, params): 
    a = actions[act] 
    if isinstance(a, tuple): 
        if a[1]=="increase": params[PARAMS.index(a[0])] += a[2]
        if a[1]=="decrease": params[PARAMS.index(a[0])] -= a[2] 
    return np.round(params, 3)



class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, obs_dim, act_dim): 
        super(CustomEnv, self).__init__() 
        self.obs_dim = obs_dim 
        self.act_dim = act_dim 
        self.action_space = spaces.Discrete(act_dim) 
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, shape=(obs_dim,), dtype=np.float32) 
        self.observation = None 
        self.timestep = 0 

    def step(self, action): 
        self.timestep += 1 
        
        """
        get model parameters (beta, gamma) from action 
        """
        # new_beta, new_gamma = calc_params_from_actions(action, self.observation[0], self.observation[1]) 
        new_params = calc_params_from_actions(action, self.observation) 

        """
        calculate next state 
        """
        self.observation = np.array(new_params, dtype=np.float32) 
        self.observation = np.clip(self.observation, OBS_LOW, OBS_HIGH) 
        
        """
        apply action to current state 
        """
        fitted_parameters = tuple(new_params) 
        t = np.linspace(0, n, n)
        res = odeint(model_deriv, y0, t, args=(N, fitted_parameters, cor_tab, nb_comp)) 
        fitted_curve = res.T 
        fitted_curve_dataframe = pd.DataFrame(columns=name_comp) 
        for i in range(nb_comp): fitted_curve_dataframe[name_comp[i]] = fitted_curve[i] 
        mae = mean_absolute_error(data['Infected'][:AMOUNT_OF_DATA], fitted_curve_dataframe['Infected'][:AMOUNT_OF_DATA])  # MAE of simulated infected data points 

        """
        calculate reward
        """
        reward = -1* (mae - MAE_MEAN) / (np.sqrt(STD)*MAE_STD)  
        
        """
        calculate done and info 
        """ 
        done = True if (self.timestep >= MAX_EP_LENGTH) else False 
        info = {"mae": mae} 
        
        return self.observation, reward, done, info 

    def reset(self, eval_guess=None):
        self.timestep = 0 
        if eval_guess: 
            self.observation = np.array(eval_guess, dtype=np.float32) 
        else: 
            self.observation = np.round(np.array(
                [np.random.uniform(OBS_LOW, OBS_HIGH) for _ in range(self.obs_dim)], 
                    dtype=np.float32), 3) 
        return self.observation 

print(NUM_PARAMS)
env = CustomEnv(obs_dim=NUM_PARAMS, act_dim=ACT_DIM) 
env.reset() 
test_env = CustomEnv(obs_dim=NUM_PARAMS, act_dim=ACT_DIM) 
test_env.reset() 
agent = PPO(state_dim=NUM_PARAMS, action_dim=ACT_DIM, hyperparams=hyperparams) 


# writer = SummaryWriter(hyperparams["logs_dir"])
log_dir = Path(os.path.join(hyperparams["logs_dir"], LOG_NAME ) )
writer = SummaryWriter(log_dir)
# log_dir = Path(hyperparams["logs_dir"])
# for i in count(0):
#     temp = log_dir/('run{}'.format(i)) 
#     if temp.exists():
#         pass
#     else:
#         writer = SummaryWriter(temp)
#         log_dir = temp
#         break

collect_mae = [] 
print("Collecting MAEs...")
for i_episode in tqdm(range(1, hyperparams["max_episodes"]+1)): 
    state = env.reset() 
    for t in range(1, hyperparams["max_cycles"]+1):
        action = np.random.randint(ACT_DIM) 
        state, reward, done, info = env.step(action) 
        collect_mae.append(info["mae"])
        if done: 
            break 
MAE_MEAN = np.mean(collect_mae)  
MAE_STD = np.std(collect_mae) 

def fit_curve(params): 
    fitted_parameters = tuple(params)
    t = np.linspace(0, n, n)
    res = odeint(model_deriv, y0, t, args=(N, fitted_parameters, cor_tab, nb_comp)) 
    fitted_curve = res.T 
    fitted_curve_dataframe = pd.DataFrame(columns=name_comp) 
    for i in range(nb_comp): fitted_curve_dataframe[name_comp[i]] = fitted_curve[i] 
    mae = mean_absolute_error(data['Infected'], fitted_curve_dataframe['Infected'])  # MAE of simulated infected data points 
    return fitted_curve, mae 

def disp(train_size, t, fit, data_nr, mae, methods, name, save_dir): 
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
    # plt.savefig(f'{save_dir}/graphs/graph_{name}_{methods}_{train_size}.png')
    Path(f'{save_dir}/graphs/').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{save_dir}/graphs/graph_{name}.png')
    plt.close(fig)

eval_obs_set = np.load(args.eval_file) 
# print(eval_obs_set) 

def my_eval(): 
    best_mae = math.inf 
    best_params = [None for _ in range(NUM_PARAMS)] 
    all_eval_ep_reward = [] 
    for eval_obs in eval_obs_set: 
        state = test_env.reset(eval_guess=list(eval_obs)) 
        eval_ep_reward = 0 

        for t in range(1, hyperparams["max_cycles"]+1):
            action = agent.select_action(state, eval=True) 
            state, reward, done, info = test_env.step(action) 
            eval_ep_reward += reward 
            if info["mae"] < best_mae: 
                best_mae = info["mae"] 
                best_params = state 
            if done: 
                break 
        all_eval_ep_reward.append(eval_ep_reward) 
    return np.mean(all_eval_ep_reward), best_mae, best_params  

for i_episode in range(1, hyperparams["max_episodes"]+1): 
    state = env.reset() 
    ep_reward = 0 
    
    for t in range(1, hyperparams["max_cycles"]+1):
        action = agent.select_action(state) 
        state, reward, done, info = env.step(action) 
        # saving reward and is_terminals
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)
        ep_reward += reward 
        if done: 
            break 
            
    if i_episode % EVAL_FREQ == 0: 
        eval_rews, best_mae, best_params = my_eval()
        writer.add_scalar("Eval Episodic Return", eval_rews, i_episode) 

    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0:
        agent.update() 

    print("Episode : {} \t\t Average Reward : {}".format(i_episode, ep_reward)) 
    writer.add_scalar("Episodic Return", ep_reward, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        print("Saving model at episode: ", i_episode) 
        agent.save(log_dir/('agent.pth')) 
        t = np.linspace(0, n, n) 
        fitted_curve, mae = fit_curve(best_params)
        disp(AMOUNT_OF_DATA, t, fitted_curve, data, mae, methods="reinforcement_learning", name="checkpoint_"+str(i_episode), save_dir=log_dir) 

eval_rews, best_mae, best_params = my_eval() 
t = np.linspace(0, n, n) 
fitted_curve, mae = fit_curve(best_params)
disp(AMOUNT_OF_DATA, t, fitted_curve, data, mae, methods="reinforcement_learning", name="final", save_dir=log_dir) 
