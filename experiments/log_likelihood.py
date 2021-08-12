import os
import sys
import yaml
import wandb
import logging
import numpy as np


import fkigp.configs as configs
import fkigp.utils as utils
from fkigp.gsgp import GsGpExp
from fkigp.kissgp import KissGpExp
from fkigp.logestimate import logdet_estimate_using_cg_variants
from fkigp.logestimate import logdet_estimate_using_lz_variants
from fkigp.gps.gpoperators import KissGpLinearOperator
from fkigp.gps.gpoperators import GsGpLinearOperator

from experiments.set_up import set_up_experiment


def run_llk_experiment(exp, options):

    sigma = exp.model.noise_covar.noise
    ntrials = options.ntrials
    nrank = options.maxiter

    if type(exp) == KissGpExp:

        t1 = utils.tic()
        W_train, K, __ = exp.model.covar_module(exp.train_x, is_kmm=True)
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

    elif type(exp) == GsGpExp:

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

    else:
        raise NotImplementedError

    return inference_time, estimate_logdet


def main(options=None):

    # Handling experiment configuration
    logging.info('Running with args %s', str(sys.argv[1:]))
    wandb.init(project="skigp")
    options = utils.get_options() if options is None else options
    wandb.config.update(options)

    # Handling log directory
    sweep_name = os.environ.get(wandb.env.SWEEP_ID, 'solo')
    output_dir = options.log_dir + '/' + sweep_name
    grid_size = -1

    if options.data_type == utils.DatasetType.PRECIPITATION:
        grid_size = np.prod(configs.get_precip_grid(options.grid_idx))
        options.log_dir = output_dir + "/rid_" + str(options.seed) \
            + "_method_" + str(options.method.value) + "_ns_" + str(options.num_samples) + "_gs_" \
            + str(grid_size)
    else:
        options.log_dir = output_dir + "/rid_" + str(options.seed) \
                           + "_method_" + str(options.method.value) + "_gs_" + str(options.grid_size)

    logging.info("Logging directory: " + options.log_dir)
    os.makedirs(options.log_dir, exist_ok=True)

    # Setting up experiment
    experiment, data_time = set_up_experiment(options)
    if grid_size < 0:
        grid_size = options.grid_size

    # Performing inference
    inference_time, logdet = run_llk_experiment(experiment, options)

    # Report results in a yaml file
    results = {
        'data_type': options.data_type.name.lower(),
        'seed': options.seed,
        'method': options.method.value,
        'num_samples': options.num_samples,
        'inf_time': float(inference_time),
        'pre_time': float(data_time),
        "error": float(np.abs(logdet - options.ref_logdet)),
    }

    if options.grid_size_f != utils.GridSizeFunc.NOT_SUPPLIED:
        results.update({'gsf': options.grid_size_f.value})
    else:
        results.update({'grid_size': int(grid_size)})

    with open(options.log_dir + "/results.yaml", 'w') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)

    logging.info("Done with experimentation!")


if __name__ == '__main__':
    main()
