import os
import sys
import yaml
import wandb
import random
import logging

import scipy
import pickle
import numpy as np
import fkigp.utils as utils
import fkigp.configs as configs

from fkigp.dataloader import DataLoader
from fkigp.gsgp import GsGpExp
from fkigp.kissgp import KissGpExp
from fkigp.gsgp import DefaultGsGpRegressionModel
from fkigp.kissgp import DefaultKissGpRegressionModel


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class QuickGsGpExp(GsGpExp):

    def load_data(self, **params):
        WT_times_W = params.get('WTW')
        WT_times_Y = params.get('WTy')
        YT_times_Y = params.get('yty')

        # Building the regression model
        grid = self.compute_grid(self.config.grid_bounds)
        self.grid = grid
        self.grid_bounds = self.config.grid_bounds
        self.model = DefaultGsGpRegressionModel(WT_times_W=WT_times_W,
                                                WT_times_Y=WT_times_Y,
                                                YT_times_Y=YT_times_Y,
                                                grid=grid,
                                                gsgp_type=self.config.gsgp_type,
                                                data_type=self.config.data_type,
                                                zero_mean=self.config.zero_mean,
                                                ard_num_dims=3,
                                                num_dims=self.config.num_dims)


class QuicKissgpExp(KissGpExp):

    def load_data(self, **params):
        W_train = params.get('W_train')
        train_y = params.get('train_y')

        # Building the regression model
        grid = self.compute_grid(self.config.grid_bounds)
        self.grid = grid
        self.grid_bounds = self.config.grid_bounds
        self.model = DefaultKissGpRegressionModel(
            W_train,
            train_y,
            zero_mean=self.config.zero_mean,
            grid=grid,
            num_dims=self.config.num_dims)


def get_radar_config(grid_idx):
    config = configs.Structdict()
    config['data_type'] = configs.DatasetType.RADAR
    config['zero_mean'] = True
    config['num_dims'] = 3
    config['grid_size'] = [item[-1] for item in configs.get_radar_grid(grid_idx)]
    config['grid_bounds'] = tuple([(item[0], item[1]) for item in configs.get_radar_grid(grid_idx)])
    return config


def set_up_gsgp_experiment(grid_idx, data_path, W_path=None):

    config = get_radar_config(grid_idx)

    print("Loading data from ... " + data_path)
    WTW = scipy.sparse.load_npz(data_path + '/WTW_train.npz')
    WTy = np.load(data_path + '/WTy_train.npz')['WTy_train']

    W_test = scipy.sparse.load_npz(data_path + '/W_test.npz')
    y_test = np.load(data_path + '/y_test.npz')['y_test']

    yty = pickle.load(open(data_path + "/norms.pkl", "rb"))[0]
    print("Done with loading dataset!")

    radar_gsgp = QuickGsGpExp(config=config, data_loader=DataLoader(config=config))

    if W_path is not None:
        W_train = scipy.sparse.load_npz(W_path + '/W_train.npz')
        radar_gsgp.WT = W_train.T.tocsr()
    radar_gsgp.load_data(WTW=WTW, WTy=WTy, yty=yty)

    radar_gsgp.build()

    return radar_gsgp, W_test, y_test


def set_up_kissgp_experiment(grid_idx, data_path):

    config = get_radar_config(grid_idx)

    print("Loading data from ... " + data_path)
    W_train = scipy.sparse.load_npz(data_path + '/W_train.npz')
    y_train = np.load(data_path + '/y_train.npz')['y_train']

    W_test = scipy.sparse.load_npz(data_path + '/W_test.npz')
    y_test = np.load(data_path + '/y_test.npz')['y_test']

    print("Done with loading dataset!")

    radar_kissgp = QuicKissgpExp(config=config, data_loader=DataLoader(config=config))
    radar_kissgp.load_data(W_train=W_train, train_y=y_train)

    return radar_kissgp, W_test, y_test


def set_up_radar_experiment(options, data_path, **kwargs):

    if options.method == utils.MethodName.GSGP:
        exp, W_test, y_test = set_up_gsgp_experiment(options.grid_idx, data_path, **kwargs)
    elif options.method == utils.MethodName.KISSGP:
        exp, W_test, y_test = set_up_kissgp_experiment(options.grid_idx, data_path)
    else:
        raise NotImplementedError

    hypers = configs.get_hypers(data_type=utils.DatasetType.RADAR, options=options)
    hypers = {
        'noise_covar.noise':  hypers['noise'],
        'covar_module.base_kernel.outputscale': hypers['outputscale'],
        'covar_module.base_kernel.base_kernel.lengthscale': hypers['lengthscale']
        if type(hypers['lengthscale']) != list else np.array(hypers['lengthscale'])
     }

    exp.model.initialize(**hypers)
    exp.print_model_params(raw=False)

    return exp, W_test, y_test


def run_inference(options, data_path):

    exp, W_test, y_test = set_up_radar_experiment(options, data_path)

    t0 = utils.tic()
    mu_grid = exp.model.predict(X=None, grid=True, verbose=True, tol=1e-2)[0]
    t0f = utils.toc(t0)
    inference_time = utils.toc_report(t0f, tag="InfGP", return_val=True)
    y_predict = (W_test*mu_grid).squeeze()

    mae_mp = compute_mae(np.ones_like(y_test)*np.mean(y_test), y_test)

    print("Mae: ", compute_mae(y_predict, y_test))
    print("Smae: ", compute_mae(y_predict, y_test)/mae_mp)
    print("Mse: ", np.mean((y_predict - y_test)**2))
    print("Rmse: ", np.sqrt(np.mean((y_predict - y_test)**2)))
    return inference_time


def compute_mae(predict_y, test_y):
    return np.mean(np.abs(predict_y.squeeze() - test_y.squeeze()))


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
    inference_time = run_inference(data_path=data_path, options=options)

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
