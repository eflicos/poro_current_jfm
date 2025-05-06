# Create input file

import numpy as np
from pathlib import Path
import yaml
import random

base_dir = '/home/ejf61/rds/hpc-work/poro_current/Ampl_range_20240604_hi_res/'

# Size of array - fixed
# !! ONLY change if edit number of inputs into poro_current.py
# Number doesn't include output dir
args = 10

Q_inj = 1
tmax = 100
tplotno = 40
nx = 500
# ny = 100
L = 7 # domain size, only works for one value of L for now
dx_init = 2**(-11)
# Types 'uniform', 'cos', 'gauss', 'channel'
perm_type = 'cos' 
# For all except 'cos', only enter one tuple val for perm_ampl,perm_mean,perm_wl
# if gen_random=True, the three below are generated randomly - do not need to set them manually
perm_ampl = [(0.01,),(0.05,),(0.1,),(0.25,),(0.5,),(0.75,),(0.9,),(0.95,),(0.99,)]
perm_mean = [(1.,)]
perm_wl = [(1.,)]


# Generate random permeability fields?
gen_random = False
num_modes = 6
num_gen = 5 # Remember, these will get stacked still

if gen_random:
    perm_ampl = []
    perm_mean = [tuple(np.ones(num_modes))] # always mean=1
    perm_wl = []
    for i in range(num_gen):
        ampl_tmp = np.zeros(num_modes)
        #mean_tmp = np.ones(num_modes) # always mean=1
        wn_tmp = np.zeros(num_modes) # wave number
        
        # Our base state
        ampl_tmp[0] = 0.5
        wn_tmp[0] = L
        for j in range(num_modes-1):
            ampl_tmp[j+1] = np.random.lognormal(np.log(0.1),.75)
            #if j%2==0:
            #    ampl_tmp[j+1] = np.random.lognormal(np.log(0.1),1)
            #else:
            #    ampl_tmp[j+1] = np.random.lognormal(np.log(0.01),1)
            wn_tmp[j+1] = random.randint(L+1,15*L)
            #wl_tmp[j+1] = random.uniform(0.05,0.9)
        
        # Check wn_tmp is unique
        wn_tmp = np.sort(wn_tmp)
        diff = wn_tmp[1:] - wn_tmp[:-1]
        while min(diff)==0:
            arg_zero = int(np.argwhere(diff==0)[0]) # just take the first one
            wn_tmp[arg_zero]=random.randint(L+1,15*L)
            wn_tmp = np.sort(wn_tmp)
            diff = wn_tmp[1:]-wn_tmp[:-1]
            
        wl_tmp = L/wn_tmp
        perm_ampl.append(tuple(ampl_tmp))
        # perm_mean.append(tuple(mean_tmp))
        perm_wl.append(tuple(wl_tmp))


# Because each tuple within perm_* should be preserved, we use the indexes to build the list of combos
ind_ampl = np.arange(len(perm_ampl))
ind_mean = np.arange(len(perm_mean))
ind_wl = np.arange(len(perm_wl))
# Build the list of all possible combinations of the above parameters
parameter_array = np.stack(np.meshgrid(Q_inj, tmax, tplotno, nx, L, dx_init, perm_type, ind_ampl, ind_mean, ind_wl)).T.reshape(-1,args)

print(f'Total number of parameter sets generated: {len(parameter_array)}')

# formatting for save
fmt = '%1.15f'

# Create base_dir
Path(base_dir).mkdir(parents=True, exist_ok=True)

# Initialise dict of all inputs
all_inputs = {}

for i in range(len(parameter_array)):
    # Set output directory
    task_id = i + 1 # b/c slurm indexes from 1, easier to adjust here than there
    output_dir = base_dir + 'run' + '{0:03d}'.format(task_id) + '/'
    #print(output_dir)
    # Create the output dirs
    Path(output_dir+'Current_Thickness/').mkdir(parents=True, exist_ok=True)
    Path(output_dir+'Current_Edge/').mkdir(parents=True, exist_ok=True)
    Path(output_dir+'Current_at_Injection/').mkdir(parents=True, exist_ok=True)
    Path(output_dir+'Permeability/').mkdir(parents=True, exist_ok=True)
    Path(output_dir+'Other/').mkdir(parents=True, exist_ok=True)
    Path(base_dir+'logs/').mkdir(parents=True, exist_ok=True)
    
    # Generate the dictionary to save
    # First, get indexes for correct tuples for perm_*
    ind_ampl_tmp = int(parameter_array[i,7])
    ind_mean_tmp = int(parameter_array[i,8])
    ind_wl_tmp = int(parameter_array[i,9])
    nx_tmp = int(parameter_array[i,3])

#    if perm_ampl[ind_ampl_tmp][0]>=0.5:
#        if perm_ampl[ind_ampl_tmp][0]>=0.9:
#            nx_tmp = 3*nx_tmp
#        else:
#            nx_tmp = 2*nx_tmp

    input_file = {
        'Q': float(parameter_array[i,0]),
        'tmax': int(parameter_array[i,1]),
        'tplotno': int(parameter_array[i,2]),
        'nx': nx_tmp,# int(parameter_array[i,3]),
        # 'ny': int(parameter_array[i,4]),
        'L': int(parameter_array[i,4]),
        'dx_init': float(parameter_array[i,5]),
        'perm_type': str(parameter_array[i,6]),
        'perm_ampl': perm_ampl[ind_ampl_tmp],
        'perm_mean': perm_mean[ind_mean_tmp],
        'perm_wl': perm_wl[ind_wl_tmp],
        'output_dir': output_dir,
        'config_file': output_dir + 'config.yml'
    }
    # Save this line to its own input file
    with open(output_dir + 'config.yml','w') as file:
        yaml.dump(input_file,file,default_flow_style=False)
    
    # Save to the dict of all inputs
    all_inputs['run' + '{0:03d}'.format(i)] = input_file

# Save all inputs to base directory
with open(base_dir + 'all_inputs.yml','w') as file:
    yaml.dump(all_inputs,file,default_flow_style=False)
