import os
import sys
import yaml
import wandb
import pickle
import logging
import numpy as np


import fkigp.configs as configs
import fkigp.utils as utils

from fkigp.cgvariants import bcg, bfcg
from fkigp.gps.gpoperators import KissGpLinearOperator
from fkigp.gps.gpoperators import GsGpLinearOperator

from experiments.set_up import set_up_experiment


def run_ski_cov_inf_experiment(exp, options):

    sigma = exp.model.noise_covar.noise

    if options.method == configs.MethodName.KISSGP:

        W_test = exp.model.covar_module(exp.test_x)[0]

        t1 = utils.tic()
        W_train, K, __ = exp.model.covar_module(exp.train_x, is_kmm=True)
        A = KissGpLinearOperator(W_train, K, sigma, dtype=W_train.dtype)
        probes = W_train * A.kmm_matmat(W_test.T.todense())   # Computing over test data points

        cov = bcg(A,
                  probes,
                  tol=options.tol,
                  maxiter=options.maxiter,
                  verbose=True)
        covf = np.dot(probes.T, cov)
        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    elif options.method == configs.MethodName.GSGP:

        W_test = exp.model.covar_module(exp.test_x)[0]

        t1 = utils.tic()
        K = exp.model.covar_module._inducing_forward()
        A_hat = GsGpLinearOperator(exp.model.WT_times_W, K, sigma, dtype=exp.model.WT_times_Y.dtype)
        r0_hat = A_hat.kmm_matmat(W_test.T.todense())

        x_diff = bfcg(A_hat,
                      r0_hat,
                      yty=np.linalg.norm(r0_hat, axis=0)**2,
                      maxiter=options.maxiter,
                      verbose=True,
                      tol=options.tol)
        covf = np.dot(r0_hat.T, x_diff)
        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    else:
        raise NotImplementedError

    if options.store_ref:
        os.makedirs(options.log_dir, exist_ok=True)
        pickle.dump(covf, open(options.log_dir + "/" + options.data_type.name.lower() + "_ski_dump.pkl", "wb"))
        return 0.0, 0.0

    # Computing l2norm
    cov_ref_path = os.environ['PRJ'] + '/data/refs/' + options.data_type.name.lower() + '_ski_dump.pkl'
    assert os.path.exists(cov_ref_path), cov_ref_path + "doesn't exists. Follow readme to generate refs."

    COV_REF = pickle.load(open(cov_ref_path, "rb"))
    l2_norm = np.linalg.norm(covf - COV_REF)
    print("L2norm: ", l2_norm)

    return inference_time, l2_norm


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
    inference_time, error = run_ski_cov_inf_experiment(experiment, options)

    # Report results in a yaml file
    results = {
        'data_type': options.data_type.name.lower(),
        'seed': options.seed,
        'method': options.method.value,
        'num_samples': options.num_samples,
        'inf_time': float(inference_time),
        'pre_time': float(data_time),
        "error": float(error),
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
