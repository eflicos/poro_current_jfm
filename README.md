# poro_current_jfm

This code solves the (dimensionless) vertically integrated gravity current equations for the thickness of a gravity current, $h$, with a spatially varying permeability field, $k$,
``` math
\frac{\partial h}{\partial t} = \nabla \dot ( k h \nabla h ),
```
on a rectangular domain $0\le x \le X$, $0 \le y \le Y$,
subject to a constant injection at $x=0$ along the width of the domain and no flux across the other three boundaries.

## Packages
It uses the following python packages. You may need to install:
* numpy
* scipy
* yaml

## Running the code
There is a default config file in the `Output` directory. To run, type:
``` bash
python3 poro_current.py --config_file='./Output/default.yml'
```

### Inputs
Inputs parameters (list below) can either be editted from the command line or in the configuration file. There are defaults within the code. The order of precedence from highest to lowest is command line, configuration file, preset defaults. All the inputs used during a run are saved separately (see Output below), so there is always a record of the values used.

Input parameters:
* config_file: location of the configuration file
* output_dir: where the output is saved (see below)
* dx_init: initial grid spacing in x
* nx: number of grid points in x
* L: width of domain in y (if using sinusoidal permeability, recommend this is an integer multiple of the largest wavelength)
* Q: injection flux (per unit length in y)
* perm_type: type of permeability field (options: 'uniform', 'cos', 'gauss', 'channel')
* perm_mean: mean value of the permeability
* perm_wl: wavelength of permeability variation
* perm_ampl: amplitude of permeability variations
* tmax: end time of the simulation
* tplotno: number of saves the code will do

The two time parameters, `tmax` and `tplotno` are used to generate target plot times. The code saves when the time goes above a plot time. We have an adaptive timestep, so sometimes the time it does save at isn't quite the same as the target time. It is the real times which are saved in the `Output/Other/` directory - see below.

#### Grid spacing
The spacing in $y$ is set using `L` and the `perm_wl`. Within the code, we prescribe a resolution of 16 grid points for the smallest permeability wavelength (set using pts_per_perm); this gives `dy`. The number of grid points is calculated by dividing the width of the domain, `L`, by `dy`. 

The initial grid spacing in $x$ is set by `dx_init` and `nx`. When the maximum height in the final 10% of grid points exceeds a threshold `small = 1e-7` (set in `check_regrid`), the regridding algorithm is called. This finds the trailing edge of the nose and then regrids up to 90% of that location. It regrids by adding neighbouring pairs of cells together and averaging their height. A maximum grid size is set using `dx_max`; cells exceeding this will no longer be regridded.

#### Generating batch input
The script `Input_generation/generate_input_values.py` can be edited and used to generate input files for a batch of runs. 

### Output
An example directory structure is shown below. Anything with a number after it is saved at the frequency set by `tplotno`. The two files `edge_mean.txt` and `edge_dev.txt` are appended to at each save with the mean and standard deviation of the edge position, respectively. The remaining files record the input parameters and things calculated from them (e.g. permability).
```
├── Current_at_Injection
│   ├── h_inj00.txt
│   ├── h_inj01.txt
│   ├── ...
├── Current_Edge
│   ├── edge00.txt
│   ├── edge01.txt
│   ├── ...
├── Current_Thickness
│   ├── h00.txt
│   ├── h01.txt
│   ├── ...
├── Other
│   ├── dx00.txt
│   ├── dx01.txt
│   ├── ...
│   ├── dy.txt
│   ├── edge_dev.txt
│   ├── edge_mean.txt
│   ├── grid_size.txt
│   ├── injection.txt
│   ├── perm_wavelength.txt
│   ├── plot_height_times.txt
│   ├── plot_times.txt
│   ├── regrid_times.txt
│   ├── regrid_range.txt
│   └── volume.txt
├── Permeability
│   └── perm.txt
├── poro_current.log
```
* `Current_at_Injection/h_injXX.txt`: the height of the current along the injection interval
* `Current_Edge/edgeXX.txt`: the $(x,y)$ co-ordinates of the edge of the current
* `Current_Thickness/hXX.txt`: the full height profile of the current
* `Other`:
    * `dxXX.txt`: the grid spacing in $x$ for that save
    * `dy.txt`: the grid spacing in $y$ (this does not change with time), saved at the start of a run
    * `edge_mean.txt`: the mean of the edge location; averaged across $y$, updated at each save time
    * `edge_dev.txt`: the standard deviation of the edge location, updated at each save time
    * `grid_size.txt`: the number of grid points saved as `nx ny`
    * `injection.txt`: the injection array, saved at the start of a run
    * `perm_wavelength.txt`: the permeability wavelength, saved at the start of a run
    * `plot_times.txt`: the times things are saved. The target times are set using the inputs `tmax` and `tplotno` and the times things are actually saved at are recorded here (usually the target and actual are the same after the first few timesteps). Updated at each save time
    * `plot_height_times.txt`: the times the height profile is saved. Within the code, it is possible to set it so that the appended files are updated every save but the full height profile only saved every X times (using `hplot_freq`). By default this is set to 1, i.e. plot_height_times is the same as plot_times
    * `regrid_times.txt`: the times at which the grid spacing is updated, updated every time the regrid function is run
    * `regrid_range.txt`: the range of indices in $x$ which are regridded, saved as `[ind_start, ind_end]`, updated every time the regrid function is run
    * `volume.txt`: the integrated volume in the current, updated at each save time. Updated at each save time
* `Permeability/perm.txt`: the permeability field, saved at the start of a run

A log file is generated. The default level is `INFO`; this can be changed in the body of the code.
