#!/usr/bin/env bash

source scripts/setup.sh
export data_dir=${PRJ_PATH}'/data'


echo 'Downloading data to '$data_dir
mkdir -p $data_dir

# Getting small datasets (solar, audio and precipitation) from google drive
tmp_data_file=$data_dir/'tmp.zip'
$gdown https://drive.google.com/uc?id=1mrFVlMdNBcLFL4XnKTcUMPAsdBqqjFrx -O $tmp_data_file
unzip $tmp_data_file -d $data_dir
rm $tmp_data_file


# Getting radar dataset -- this will dump data in ${PRJ_PATH}'/data/radar/'
$py -m experiments.radar_processing --download_radar