from experiments.mean_inference import main as main_inference
from experiments.log_likelihood import main as log_likelihood

import fkigp.configs as configs
import fkigp.utils as utils


def main():

    options = utils.get_options()

    if options.experiment_type == configs.ExperimentType.MEAN_INF:
        options['maxiter'] = 1000
        options['tol'] = 1e-1
        main_inference(options=options)

    elif options.experiment_type == configs.ExperimentType.LOGDET:

        options['maxiter'] = 400
        options['tol'] = 1e-1  # For reproducing, Figure 7 panel 2 change this to 1e-2.
        options['ref_logdet'] = 1501502.031721752  # computed using higher number of iterations for CG
        log_likelihood(options=options)


if __name__ == '__main__':
    main()
