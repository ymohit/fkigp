import os
import sys
import yaml
import wandb
import pickle
import logging
import numpy as np
import fastmat as fm

import fkigp.configs as configs
import fkigp.utils as utils
from fkigp.solvers import chol_trid, chol_trid_solve
from fkigp.lzvariants import bsla, bfsla1
from fkigp.gps.gpoperators import KissGpLinearOperator
from fkigp.gps.gpoperators import GsGpLinearOperator

from experiments.set_up import set_up_experiment


MAX_LANCZOS_RANK = 10000


def run_approx_cov_inf_experiment(exp, options):

    if exp.config.data_type not in [configs.DatasetType.SOUND, configs.DatasetType.PRECIPITATION]:
        nrank = min(configs.DEFAULT_LANCZOS_RANK, options.maxiter)
        print("\n\n\nLanczos rank is reduced to ", nrank, "  from ", options.maxiter)
        print("\n\n")
    else:
        nrank = options.maxiter

    sigma = exp.model.noise_covar.noise
    if exp.config.data_type == configs.DatasetType.SOUND:
        max_num_test_vectors = 1000
    else:
        max_num_test_vectors = 50

    if exp.test_x.shape[0] > max_num_test_vectors:
        W_test = exp.model.covar_module(exp.test_x[:min(max_num_test_vectors, exp.test_x.shape[0]), :])[0]
    else:
        W_test = exp.model.covar_module(exp.test_x)[0]

    if options.method == configs.MethodName.KISSGP:

        t1 = utils.tic()
        W_train, K, __ = exp.model.covar_module(exp.train_x, is_kmm=True)

        A = KissGpLinearOperator(W_train, K, sigma, dtype=W_train.dtype)
        probes = A.kmm_matmat(W_train.T.todense()).T
        v = np.mean(probes, axis=1)
        Q, T = bsla(A, v, k=nrank)
        T_diag = T.diagonal()
        T_subdiag = T.diagonal(1)
        L_diag, L_subdiag = chol_trid(T_diag, T_subdiag)
        R = A.kmm_matmat(A.WT * Q)
        Rprime = chol_trid_solve(L_diag, L_subdiag, R.T).T

        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    elif options.method == configs.MethodName.GSGP:

        t1 = utils.tic()
        K = exp.model.covar_module._inducing_forward()
        probes = fm.Kron(*K).getArray().real if len(K) > 1 else K[0].getArray().real
        v = np.mean(probes, axis=1)

        A_hat = GsGpLinearOperator(exp.model.WT_times_W, K, sigma, dtype=exp.model.WT_times_Y.dtype)
        Q, T = bfsla1(A_hat, v, k=nrank)
        T_diag = T.diagonal()
        T_subdiag = T.diagonal(1)
        L_diag, L_subdiag = chol_trid(T_diag, T_subdiag)
        R = A_hat.kmm_matmat(A_hat.WTW * Q)
        Rprime = chol_trid_solve(L_diag, L_subdiag, R.T).T
        t1f = utils.toc(t1)
        inference_time = utils.toc_report(t1f, tag="InfGP", )

    else:
        raise NotImplementedError

    # Computing l2norm
    cov_ref_path = os.environ['PRJ'] + '/data/refs/' + options.data_type.name.lower() + '_ski_dump.pkl'
    assert os.path.exists(cov_ref_path), cov_ref_path + "doesn't exists. Follow readme to generate refs."

    COV_REF = pickle.load(open(cov_ref_path, "rb"))[-1]
    predicted_cov = np.dot(W_test * R, (W_test * Rprime).T)
    l2_norm = np.linalg.norm(predicted_cov - COV_REF)
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
    inference_time, error = run_approx_cov_inf_experiment(experiment, options)

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
