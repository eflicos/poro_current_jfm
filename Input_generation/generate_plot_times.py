# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:35:53 2019

@author: ajb278
"""
import numpy as np

# Filepath of Input directory
output_dir = '/mnt/c/Users/Emily/Documents/PhD/Scripts/Python/poro_current/Input/'

def main():
    
    # Options
    print('Choose time spacing:')
    choice = int(input('1) Linear or 2) Logarithmic \n'))

    if choice == 1:
        print('Linear time spacing')
        t_end = float(input('End time:\n'))
        n_plot = int(input('Number of plots:\n'))
        plot_times = lin_dt(0,t_end,n_plot)
    elif choice == 2:
        print('Logarithmic time spacing')
        t_start = 1e-6
        t_end = float(input('End time:\n'))
        n_plot = int(input('Number of plots:\n'))
        plot_times = np.logspace(np.log10(t_start), np.log10(t_end), n_plot)
    else:
        print('Incorrect choice, please try again')

    # Print and save output
    print(plot_times)
    np.savetxt(output_dir + 'Input_plot_times.txt',plot_times)
    
def lin_dt(t_start,t_end,n_plot):
    plot_times = np.linspace(t_start,t_end,n_plot)
    return plot_times

def log_dt(t_start,t_end,n_plot):
    plot_times = np.logspace(np.log10(t_start), np.log10(t_end), n_plot)
    return plot_times
    
def log_dt_multimodal(t_start,t_end,n_plot,perm_wl):
    perm_wl_min = min(perm_wl)
    perm_wl_max = max(perm_wl)
    
    t1 = (perm_wl_min/2)**(3/2)
    t2 = (perm_wl_max/2)**(3/2)
    
    times1 = np.logspace(np.log10(t_start), np.log10(t1), int(n_plot/3),endpoint=False)
    times2 = np.logspace(np.log10(t1), np.log10(t2), int(n_plot/3),endpoint=False)
    times3 = np.logspace(np.log10(t2), np.log10(t_end), n_plot-(len(times1)+len(times2)),endpoint=True)
    
    plot_times = np.concatenate((times1,times2,times3))
    return plot_times

if __name__ == '__main__':
    main()