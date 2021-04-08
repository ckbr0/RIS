#!/usr/bin/python3 -u

import sys
import os
import glob
import argparse

from sklearn.model_selection import train_test_split

from utils import get_data_from_info
from train_workflow import TrainingWorkflow
from train import Training

def main(parse_args=False):
    data_check = False
    if parse_args:
        parser = argparse.ArgumentParser(description='RIS.')
        parser.add_argument("--data_check", help="run data check", action="store_true")
        args = parser.parse_args()
        if args.data_check:
            data_check = True
            print("running data check...")

    # Osnovne datoteke
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(src_dir, '..', 'data'))
    out_dir = os.path.abspath(os.path.join(src_dir, '..', 'out'))
    cache_dir = os.path.abspath(os.path.join(src_dir, '..', 'cache'))

    hackathon_dir = os.path.join(data_dir, 'HACKATHON')
    # Naloži train hackathon podatje
    with open(hackathon_dir + "/train.txt", 'r') as fp:
        train_info = [entry.strip().split(',') for entry in fp.readlines()]
    #path_to_images = os.path.join(hackathon_dir, 'images', 'train')
    #path_to_segs = os.path.join(hackathon_dir, 'segmentations', 'train')
    #train_data = get_data_from_info(path_to_images, path_to_segs, train_info)

    # Naloži druge podatke
    # TODO: Podatki iz vaj
    extra_valid_info = []
    
    # Naloži podatke za končni test
    path_to_images = os.path.join(hackathon_dir, 'images', 'test')
    path_to_segs = os.path.join(hackathon_dir, 'segmentations', 'test')
    images = glob.glob(os.path.join(path_to_images, '*'))
    end_test_info = [(os.path.basename(image),-1) for image in images]
    end_test_info = get_data_from_info(path_to_images, path_to_segs, end_test_info)

    # Train data razdelimo na train, validation in testing sete
    # TODO: Ali mora biti to deterministično? Ali se lahko spreminja iz runa v run?

    # Naiven split, determinističen, ker je random_state določen
    train_split, test_info = train_test_split(train_info, test_size=0.2, shuffle=True, random_state=42)
    train_info, valid_info = train_test_split(train_split, test_size=0.2, shuffle=True, random_state=43)
    #valid_info = list(zip(valid_info, extra_valid_info))

    print(f'train len: {len(train_info)}, valid len: {len(valid_info)}, test len: {len(test_info)}')

    # Nastavimo hiperparametre v slovarju
    hyperparameters = {}
    hyperparameters['learning_rate'] = 0.2e-3 # learning rate
    hyperparameters['weight_decay'] = 0.0001 # weight decay
    hyperparameters['total_epoch'] = 6 # total number of epochs
    hyperparameters['multiplicator'] = 0.95 # each epoch learning rate is decreased on LR*multiplicator
    hyperparameters['batch_size'] = 1
    hyperparameters['validation_epoch'] = 1 # Only perform validations if current epoch is greater or equal validation_epoch
    hyperparameters['validation_interval'] = 1
    hyperparameters['H'] = 1500
    hyperparameters['L'] = -600

    if os.name == 'nt':
        num_workers = 0
    else:
        num_workers = 0

    training = Training(data_dir, hackathon_dir, out_dir, cache_dir, 'model_ct', num_workers=num_workers, cuda=True)

    training.train(train_info, valid_info, hyperparameters, run_data_check=data_check)
    
if __name__ == '__main__':
    main(True)

