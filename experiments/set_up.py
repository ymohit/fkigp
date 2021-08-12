import copy
import random
import numpy as np
import fkigp.utils as utils
import fkigp.configs as configs

from fkigp.gsgp import GsGpExp
from fkigp.kissgp import KissGpExp
from fkigp.dataloader import DataLoader


def set_up_experiment(options):

    # Setup random seed
    random.seed(options.seed)
    np.random.seed(options.seed)

    # Setting up the experiment skeleton
    if options.data_type == configs.DatasetType.PRECIPITATION:
        grid_size = configs.get_precip_grid(idx=options.grid_idx)
        config = configs.Structdict()
        config['data_type'] = utils.DatasetType.PRECIPITATION
        config['num_dims'] = 3
        config['grid_size'] = copy.copy(grid_size)
        data_reader = DataLoader(config=options)
    else:
        data_reader = DataLoader(config=options)

    if options.method == configs.MethodName.KISSGP:
        experiment = KissGpExp(config=options, data_loader=data_reader)

    elif options.method == configs.MethodName.GSGP:
        experiment = GsGpExp(config=options, data_loader=data_reader)

    else:
        raise NotImplementedError

    # Supply test scenario -- required only for per iteration results
    experiment.data_loader.config["one_dim_num_points"] = options.num_samples
    if options.sigma > 0:
        experiment.data_loader.config['one_dim_noise_level'] = options.sigma

    # Setting grid size
    num_points = options.num_samples
    if options.grid_idx < 0 and experiment.config['grid_size'] < 0:
        grid_size = configs.get_grid_size(
            num_points=num_points, grid_size_f=options.grid_size_f, data_type=experiment.config.data_type
        )
        experiment.config['grid_size'] = grid_size

    # Sampling or reading data
    experiment.sample_data()  # this is to cut time used for synthetic data creation

    # Processing data
    t0 = utils.tic()
    experiment.load_data()
    t0f = utils.toc(t0)
    data_time = utils.toc_report(t0f, tag="DataGP", return_val=True)

    # Build experiment
    experiment.build()

    # Dealing with hyper-parameters
    hypers = configs.get_hypers(data_type=options.data_type, options=options)
    hypers = {
        'noise_covar.noise':  hypers['noise'],
        'covar_module.base_kernel.outputscale': hypers['outputscale'],
        'covar_module.base_kernel.base_kernel.lengthscale': hypers['lengthscale']
        if type(hypers['lengthscale']) != list else np.array(hypers['lengthscale'])
     }
    experiment.init_params(hypers=hypers)
    experiment.print_model_params(raw=False)

    return experiment, data_time
