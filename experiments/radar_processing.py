import os
import yaml
import tqdm
import pyart
import pickle
import scipy

import numpy as np

import wsrlib
from wsrlib import get_volumes, z_to_refl, idb

import fkigp.utils as utils
import fkigp.configs as configs

from fkigp.gridutils import get_basis


TRAIN_FRAC = 0.999
DEFAULT_RADAR_GRID_ORIGIN = [42.0, -75.0]
DEFAULT_RADAR_GRID_BOUNDS = ((500, 3000), (-700000, 700000), (-700000, 700000))  # +- 700km
DEFAULT_RADAR_GRID_COORDINATES = [
    (36, 48, 100),  # latitude
    (-82, -64, 100),  # longitude
    (500, 3000, 6)  # elevation
]

DEFAULT_Z_MAX = 3000
DEFAULT_DBZ_VALUE = -30

RADAR_DATASET_PATH = configs.PRJ_PATH + 'data/radar'
os.makedirs(RADAR_DATASET_PATH, exist_ok=True)


def download_data():

    # downloading for north-eastern data
    print("Downloading NE data ...")
    file_name_list = os.environ['PRJ'] + 'configs/radar_north_eastern.yaml'
    NE_DIR = RADAR_DATASET_PATH + '/ne/'
    os.makedirs(NE_DIR, exist_ok=True)
    file_names = yaml.load(open(file_name_list))
    for file_name in tqdm.tqdm(file_names):
        wsrlib.get_s3(file_name, localfile=NE_DIR + file_name + '.gz')

    # downloading for north-eastern data
    print("Downloading Entire US data ...")
    file_name_list = os.environ['PRJ'] + 'configs/radar_entire_us.yaml'
    ENTIRE_US_DIR = RADAR_DATASET_PATH + '/entire_us/'
    os.makedirs(ENTIRE_US_DIR, exist_ok=True)
    file_names = yaml.load(open(file_name_list))
    for file_name in tqdm.tqdm(file_names):
        wsrlib.get_s3(file_name, localfile=ENTIRE_US_DIR + file_name + '.gz')


def get_data_poionts(radar, default_val=-30, zmax=3000):
    LON, LAT, ALT, DBZ = get_volumes(radar, field='reflectivity', coords='geographic')

    inds = ALT <= zmax
    LON, LAT, ALT, DBZ = LON[inds], LAT[inds], ALT[inds], DBZ[inds]

    # Transform response variable
    DBZ[np.isnan(DBZ)] = default_val
    ETA, _ = z_to_refl(idb(DBZ))
    y = ETA ** (1 / 7)  # Nusbaummer paper
    X = np.vstack([LAT, LON, ALT]).T  # Dimension order: lat, lon alt
    return X, y


def pre_process(method, grid_idx, entire_us=False):

    zmax = DEFAULT_Z_MAX
    grid = configs.get_radar_grid(idx=grid_idx)
    num_grid_points = np.product([item[-1] for item in grid])

    # Read all of the radars into file
    if entire_us:
        data_dirpath = RADAR_DATASET_PATH + "/entire_us"
    else:
        data_dirpath = RADAR_DATASET_PATH + "/ne"
    files = os.listdir(data_dirpath)

    # Creating directory for processed files
    output_data_path = data_dirpath + "_processed/" + method.name.lower() + "_grid_" + str(grid_idx)
    os.makedirs(output_data_path, exist_ok=True)

    if method == utils.MethodName.GSGP:

        t0 = utils.tic()
        WTW_train, WTy_train, yty_train, n_train, total_nnz = 0, 0, 0, 0, 0
        W_test, y_test, n_test = [], [], 0

        print("\n\nProcessing data ...\n\n")

        print("Reading data ...\n\n")

        for scan in files:
            filename = '%s' % (scan)
            print('Reading %s' % (filename))
            try:
                print("File name path: ", data_dirpath + "/" + filename.split("/")[-1])
                radar = pyart.io.read_nexrad_archive(data_dirpath + "/" + filename.split("/")[-1])

            except IOError:
                pass

            print('Processing %s' % (radar.metadata['instrument_name']))

            X, y = get_data_poionts(radar, zmax=zmax)

            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]

            ntrain = int(TRAIN_FRAC * len(X))
            W_train = get_basis(X[:ntrain], grid)
            y_train = y[:ntrain]
            WT_train = W_train.T.tocsr()
            total_nnz += len(W_train.nonzero()[0])

            WTW_train += WT_train * W_train
            WTy_train += WT_train * y_train
            yty_train += y_train.T @ y_train
            n_train += ntrain

            W_test += get_basis(X[ntrain:], grid),
            y_test += y[ntrain:],
            n_test += len(X) - ntrain

        t0f = utils.toc(t0)
        pre_time = utils.toc_report(t0f, tag="DataGP", return_val=True)

        m_logm = num_grid_points * np.log2(num_grid_points)
        print("NumPoints:", n_train)
        print("NumTestPoints:", n_test)
        print("Expected speed up over SKI: ", (2 * total_nnz + m_logm) / (len(WTW_train.nonzero()[0]) + m_logm))

        W_test = scipy.sparse.vstack(W_test)
        y_test = np.hstack(y_test)
        scipy.sparse.save_npz(output_data_path + '/WTW_train.npz', WTW_train)
        np.savez(output_data_path + '/WTy_train.npz', WTy_train=WTy_train)
        scipy.sparse.save_npz(output_data_path + '/W_test.npz', W_test)
        np.savez(output_data_path + '/y_test.npz', y_test=y_test)
        pickle.dump((yty_train, n_train, n_test), open(output_data_path + "/norms.pkl", "wb"))

        # Report results in a yaml file
        results = {
            'n_train': n_train,
            'n_test': n_test,
            'method': method.value,
            'pre_time': float(pre_time),
            'grid_size': num_grid_points
        }
        with open(output_data_path + "/stats.yaml", 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

    elif method == utils.MethodName.KISSGP:

        print("Reading data ...\n\n")
        radars = []
        for scan in files:
            filename = '%s' % (scan)
            print('Reading %s' % (filename))
            try:
                print("File name path: ", data_dirpath + "/" + filename.split("/")[-1])
                radars.append(pyart.io.read_nexrad_archive(data_dirpath
                                                           + "/" + filename.split("/")[-1]))
            except IOError:
                pass

        t0 = utils.tic()
        W_train, y_train, n_train, n_test = [], [], 0, 0
        W_test, y_test = [], []

        print("\n\nProcessing data ...\n\n")

        for radar in radars:

            print('Processing %s' % (radar.metadata['instrument_name']))
            X, y = get_data_poionts(radar, zmax=zmax)
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]

            ntrain = int(TRAIN_FRAC * len(X))

            W_train += get_basis(X[:ntrain], grid),
            y_train += y[:ntrain],
            n_train += ntrain

            W_test += get_basis(X[ntrain:], grid),
            y_test += y[ntrain:],
            n_test += len(X) - ntrain

        W_train = scipy.sparse.vstack(W_train)
        W_test = scipy.sparse.vstack(W_test)
        y_train = np.hstack(y_train)
        y_test = np.hstack(y_test)
        t0f = utils.toc(t0)
        pre_time = utils.toc_report(t0f, tag="DataGP", return_val=True)
        print("NumPoints:", n_train)
        print("NumTestPoints:", n_test)

        scipy.sparse.save_npz(output_data_path + '/W_train.npz', W_train)
        scipy.sparse.save_npz(output_data_path + '/W_test.npz', W_test)
        np.savez(output_data_path + '/y_train.npz', y_train=y_train)
        np.savez(output_data_path + '/y_test.npz', y_test=y_test)

        # Report results in a yaml file
        results = {
            'n_train': n_train,
            'n_test': n_test,
            'method': method.value,
            'pre_time': float(pre_time),
            'grid_size': num_grid_points
        }
        with open(output_data_path + "/stats.yaml", 'w') as outfile:
            yaml.dump(results, outfile, default_flow_style=False)

    else:
        raise NotImplementedError

    print("Pre-processing time: ", pre_time)

    return


def main():

    options = utils.get_options()

    if options.download_radar:
        download_data()
        return
    pre_process(options.method, options.grid_idx, entire_us=options.entire_us)


if __name__ == '__main__':
    main()
