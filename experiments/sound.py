from experiments.mean_inference import main as main_inference
from experiments.log_likelihood import main as log_likelihood
from experiments.ski_covariance import main as ski_covariance
from experiments.ski_approx_covariance import main as ski_approx_covariance

import fkigp.configs as configs
import fkigp.utils as utils


def main():

    options = utils.get_options()

    if options.experiment_type == configs.ExperimentType.MEAN_INF:

        options['maxiter'] = 1000
        options['tol'] = 1e-2
        main_inference(options=options)

    elif options.experiment_type == configs.ExperimentType.LOGDET:

        options['maxiter'] = 100
        options['tol'] = 1e-2
        options['ref_logdet'] = -517495.271  # computed using higher number of iterations for CG
        log_likelihood(options=options)

    elif options.experiment_type == configs.ExperimentType.SKI_COV:

        options['maxiter'] = 100
        options['tol'] = 1e-2
        ski_covariance(options=options)

    elif options.experiment_type == configs.ExperimentType.APPROX_COV:
        options['maxiter'] = 500
        options['tol'] = 1e-2
        ski_approx_covariance(options=options)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
