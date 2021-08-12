import os
import sys
import yaml
import wandb
import random
import logging

import numpy as np
import fkigp.utils as utils
import fkigp.configs as configs

from fkigp.gps.gpoperators import GsGpLinearOperator
from fkigp.gps.gpoperators import KissGpLinearOperator
from fkigp.logestimate import logdet_estimate_using_lz_variants
from fkigp.logestimate import logdet_estimate_using_cg_variants

from experiments.radar_inference import set_up_radar_experiment


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def run_logdet_computation(options, data_path, W_path=None):

    ntrials = options.ntrials
    nrank = options.maxiter

    exp, __, __ = set_up_radar_experiment(options, data_path, W_path=W_path)
    sigma = exp.model.noise_covar.noise

    if options.method == utils.MethodName.GSGP:

        t1 = utils.tic()
        K = exp.model.covar_module._inducing_forward()
        A_hat = GsGpLinearOperator(exp.model.WT_times_W, K, sigma, dtype=exp.model.WT_times_Y.dtype)

        if options.variant == 1:
            estimate_logdet = logdet_estimate_using_lz_variants(
                A_hat, WT=exp.WT, trials=ntrials,
                rank=nrank, verbose=True, dump=options.dump, dump_path=options.log_dir)[0]
        elif options.variant == 2:
            estimate_logdet = logdet_estimate_using_cg_variants(
                A_hat, WT=exp.WT, trials=ntrials, tol=options.tol,
                rank=nrank, verbose=True)[0]
        else:
            raise NotImplementedError

        print("Estimated log-det: ", estimate_logdet)
        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    elif options.method == utils.MethodName.KISSGP:

        t1 = utils.tic()
        W_train = exp.model.W
        K = exp.model.covar_module._inducing_forward()
        A = KissGpLinearOperator(W_train, K, sigma, dtype=W_train.dtype)

        if options.variant == 1:
            estimate_logdet = logdet_estimate_using_lz_variants(
                A, trials=ntrials,
                rank=nrank, verbose=True,)[0]
        elif options.variant == 2:
            estimate_logdet = logdet_estimate_using_cg_variants(
                A, trials=ntrials, tol=options.tol,
                rank=nrank, verbose=True)[0]
        else:
            raise NotImplementedError

        print("Estimated log-det: ", estimate_logdet)

        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    else:
        raise NotImplementedError

    return inference_time


def main(options=None):

    # Handling experiment configuration
    logging.info('Running with args %s', str(sys.argv[1:]))
    wandb.init(project="skigp")
    options = utils.get_options() if options is None else options
    wandb.config.update(options)

    # Setup random seed
    random.seed(options.seed)
    np.random.seed(options.seed)

    # Handling log directory
    sweep_name = os.environ.get(wandb.env.SWEEP_ID, 'solo')
    output_dir = options.log_dir + '/' + sweep_name
    options.log_dir = output_dir + "/rid_" + str(options.seed) \
        + "_method_" + str(options.method.value) + "_gs_" + str(options.grid_idx)

    logging.info("Logging directory: " + options.log_dir)
    os.makedirs(options.log_dir, exist_ok=True)

    # Setting up experiment and run inference
    RADAR_DATASET_PATH = configs.PRJ_PATH + 'data/radar'
    if options.entire_us:
        data_dirpath = RADAR_DATASET_PATH + "/entire_us_processed"
    else:
        data_dirpath = RADAR_DATASET_PATH + "/ne_processed"
    data_path = data_dirpath + '/' + options.method.name.lower() + "_grid_" + str(options.grid_idx)
    W_path = data_dirpath + '/kissgp_grid_' + str(options.grid_idx)

    inference_time = run_logdet_computation(data_path=data_path, options=options, W_path=W_path)

    # Report results in a yaml file
    results = {
        'data_type': options.data_type.name.lower(),
        'seed': options.seed,
        'method': options.method.value,
        'inf_time': float(inference_time),
        'grid_idx': int(options.grid_idx)
    }
    with open(options.log_dir + "/results.yaml", 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    logging.info("Done with experimentation!")


if __name__ == '__main__':
    main()
