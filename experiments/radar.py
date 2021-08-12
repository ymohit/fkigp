from experiments.radar_inference import main as main_inference
from experiments.log_likelihood import main as log_likelihood

import fkigp.configs as configs
import fkigp.utils as utils


def main():

    options = utils.get_options()

    if options.experiment_type == configs.ExperimentType.MEAN_INF:
        options['maxiter'] = 40
        options['tol'] = 1e-1
        main_inference(options=options)

    elif options.experiment_type == configs.ExperimentType.LOGDET:

        options['maxiter'] = 60
        options['tol'] = 1e-1  # For reproducing, Figure 5 panel 2 change this to 1e-2.
        log_likelihood(options=options)


if __name__ == '__main__':
    main()
