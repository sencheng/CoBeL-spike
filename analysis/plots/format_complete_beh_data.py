import pandas as pd

data_path = "complete_beh_data.dat"

df = pd.read_csv(data_path)

df = df.drop(['Unnamed: 0', 'index'], axis=1)

df = df.rename(
    columns={
        'n_pc': 'p_ncols', 
        'sigma_pc': 'p_sigma',
        'max_fr': 'p_max_fr',
        'reward_rad': 'goal_size1',
        'tr_dur': 'tr_duration',
        'traj_len': 'traj_lentgh'
    }
)

df['goal_x'] = -1
df['goal_y'] = -1
df['num_goal'] = -1
df['proximity_min'] = -1
df['proximity_mean'] = -1
df['DTW_opt'] = -1
df['trial_switch'] = -1
df['with_trial_switch'] = -1
df['DTW'] = -1
df['fr_noise'] = 0

df.to_csv("complete_beh_data_formatted.csv", sep='\t')
