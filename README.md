
# Fast Kernel Interpolation For Gaussian Processes
This repository hosts code for our [AISTATS-2021 paper](http://proceedings.mlr.press/v130/yadav21a.html).


## Quick set-up and examples 

### Install requirements

```setup
conda create -n fkigp python=3.7
source activate fkigp
pip install -r requirements.txt
```

### (Optional) Tests/sanity visual checks:
```
tests/modules.ipynb -- show case basic necessary modules along with expected results.  
tests/kissgp.ipynb -- performs inference on three small datasets that validates the correctness of our implementation. 
tests/gsgp.ipynb -- replicate KISSGP inference for above problems while achiving the same accuracy. 
```

### Examples/demos: 
* Fast per-iteration complexity: ipynb/per_iteration.ipynb 
* SKI based GP inference via sufficient statistics: ipynb/ski_using_sufficient_stats.ipynb

## Experiments and reproducing results
All figures in the paper can be reproduced via [ipynb](./ipynb).

### Configure and get datasets
* Set python environment and project paths in scripts/setup.sh. 
* Set os.environ['PRJ'] path in fkigp/configs.py 
* Run bash scripts/get_data.sh and its expected outcome is following files/*

```
data/
├── precip_data.mat 
├── audio_data.mat
├── solar_data.txt
├── radar
│   ├── entire_us
│   ├── ne
```

### Generic experimental procedure:

* Step 1: Load paths at head node.
```
source scripts/setup.sh
```

* Step 2: Run $wandb sweep configs/*.yaml. For example, 

```
$wandb sweep configs/per_iteration_sweep.yaml 
```

This gives output like:

```
wandb: Creating sweep from: configs/per_iteration_sweep.yaml
wandb: Created sweep with ID: n25yz7k8
wandb: View sweep at: https://app.wandb.ai/ymohit/fkigp/sweeps/n25yz7k8
wandb: Run sweep agent with: wandb agent ymohit/fkigp/n25yz7k8
```

* Step 3: Then launch sweep on slurm job as follows:

```
# bash scripts/launch_sweep.sh sweepId numMachines numCpuPerMachine mbRAMperMachine
bash scripts/launch_sweep.sh ymohit/fkigp/n25yz7k8 10 1 5000
```

* Step 4: Collect data from logs/sweep_id for sweep_id. This data can contains results. 

<b>All figures in the paper can be reproduced via [ipynb](./ipynb).</b>

### Results on per-iteration complexity (i.e., Figure 3) 

* Step 1: Follow above mentioned procedure with configs/per_iteration_sweep.yaml.
* Step 2: Use ipynb/per_iteration_results.ipynb to visualize results.

### Results on sound (i.e., Figure 4)

* Step 1: Produce ref SKI covariance reference matrix as follows: 
```eval
source setup.sh
$py -m experiments.ski_covariance --store_ref --data_type 2 --grid_size 60000 --method 1 --tol 1e-8 --maxiter 2000
```
* Step 2: Follow above mentioned procedure with configs/sound_sweep.yaml.
* Step 3: Use ipynb/sounds.ipynb to visualize results.

### Results on radar (i.e., Figure 5)
* Step 1: (Pre-processing stage:) Follow above mentioned procedure with configs/radar_preprocess_sweep.yaml.
* Step 2: (Inference stage:) Follow above mentioned procedure with configs/radar_sweep.yaml.
* Step 3: Use ipynb/radar.ipynb to visualize results.


### Results on precipitation (i.e., Figure 7)
* Step 1: Follow above mentioned generic procedure with configs/precipitation_sweep.yaml.
* Step 2: Use ipynb/precipitation.ipynb to visualize results.

### Results for `very large n' case

* To process the dataset for the entire us case: 
```eval
source scripts/setup.sh
$py -m experiments.radar_processing --entire_us --method 2 --grid_idx 10
```

* To run inference for the entire us case: 
```eval
source scripts/setup.sh
$py -m experiments.radar_inference --entire_us --maxiter 100 --tol 1e-3 --grid_idx 2 --method 2
```


## License
Apache 2.0 

