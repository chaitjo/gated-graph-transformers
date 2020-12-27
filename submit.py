import os
import torch
import numpy as np
import argparse


if __name__ == "__main__":
    # Experiment settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='Dataset name (default: ogbg-molhiv)')
    parser.add_argument('--expt', type=str, default="debug",
                        help='Name of experiment logs/results folder')
    args = parser.parse_args()
    
    expt_args = []
    model = []
    total_param = []
    BestEpoch = []
    Validation = []
    Test = []
    Train = []
    BestTrain = []

    for root, dirs, files in os.walk(f'logs/{args.dataset}/{args.expt}/'):
        if 'results.pt' in files:
            results = torch.load(os.path.join(root, 'results.pt'))
            expt_args.append(results['args'])
            model.append(results['model'])
            total_param.append(results['total_param'])
            BestEpoch.append(results['BestEpoch'])
            Validation.append(results['Validation'])
            Test.append(results['Test'])
            Train.append(results['Train'])
            BestTrain.append(results['BestTrain'])

    print(f'Test performance: {np.mean(Test)} +- {np.std(Test)}')
    print(f'Validation performance: {np.mean(Validation)} +- {np.std(Validation)}')
    print(f'Train performance: {np.mean(Train)} +- {np.std(Train)}')
    print(f'Total parameters: {int(np.mean(total_param))}')
