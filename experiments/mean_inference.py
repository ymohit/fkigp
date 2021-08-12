import os
import sys
import yaml
import wandb

import logging
import numpy as np
import fkigp.configs as configs
import fkigp.utils as utils
from experiments.set_up import set_up_experiment

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


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

    if options.grid_size_f != utils.GridSizeFunc.NOT_SUPPLIED:
        options.log_dir = output_dir + "/rid_" + str(options.seed) \
            + "_method_" + str(options.method.value) + "_ns_" + str(options.num_samples) + "_gsf_" \
            + str(options.grid_size_f.value)
    elif options.data_type == utils.DatasetType.PRECIPITATION:
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
    dump = []
    t1 = utils.tic()

    if options.data_type == configs.DatasetType.SINE:
        error = experiment.compute_mae(maxiter=options.maxiter, verbose=True, dump=dump)
        t1f = utils.toc(t1)

    elif options.data_type == configs.DatasetType.SOUND:
        error = experiment.compute_smae(maxiter=options.maxiter, verbose=True, dump=dump)
        t1f = utils.toc(t1)

    elif options.data_type == configs.DatasetType.PRECIPITATION:
        predict_y = experiment.model.predict(experiment.test_x, verbose=True, tol=options.tol, maxiter=800, dump=dump)
        t1f = utils.toc(t1)
        error = np.mean(np.abs(predict_y[0].squeeze() - experiment.test_y.squeeze()))   # computing mAE
    else:
        raise NotImplementedError

    iter_count = dump[0]
    inference_time = utils.toc_report(t1f, tag="InfGP", return_val=True)

    # Report results in a yaml file
    results = {
        'data_type': options.data_type.name.lower(),
        'seed': options.seed,
        'method': options.method.value,
        'num_samples': options.num_samples,
        'inf_time': float(inference_time),
        'pre_time': float(data_time),
        "error": float(error),
        "num_iters": iter_count
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
