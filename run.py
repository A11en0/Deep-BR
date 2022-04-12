# -*- coding: UTF-8 -*-
import os
from functools import reduce

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from models import Model
from train import train, test
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data_v1, init_random_seed


def run(args, save_dir, file_name):
    print('*' * 30)
    print('dataset:\t', args.DATA_SET_NAME)
    print('optimizer:\t Adam')
    print('*' * 30)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = save_dir + file_name

    features, labels, idx_list = load_mat_data_v1(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    writer = SummaryWriter()
    fold_list = []
    rets = np.zeros((Fold_numbers, 7))
    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)
        train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
            features, labels, idx_list, TEST_SPLIT_INDEX, args)

        # load views features and labels
        views_dataset = ViewsDataset(train_features, train_partial_labels, device)
        views_data_loader = DataLoader(views_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        view_code_list = list(train_features.keys())
        view_feature_nums_list = [train_features[code].shape[1] for code in view_code_list]
        feature_dim = reduce(lambda x, y: x + y, view_feature_nums_list)
        label_nums = train_labels.shape[1]

        # load model
        model = Model(feature_dim, label_nums, device, args).to(device)

        # training
        loss_list = train(model, device, views_data_loader, args, loss_coefficient,
                     train_features, train_partial_labels, test_features, test_labels, fold=1)
        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, device, is_eval=True, args=args)

        for i, m in enumerate(metrics_results):
            rets[fold][i] = m[1]

    print("\n------------summary--------------")
    means = np.mean(rets, axis=0)
    stds = np.std(rets, axis=0)
    metrics = ['hamming_loss', 'avg_precision', 'one_error', 'ranking_loss', 'coverage', 'macrof1', 'microf1',]
    with open(save_name, "w") as f:
        for i, _ in enumerate(means):
            print("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics[i], means=means[i], std=stds[i]))
            f.write("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics[i], means=means[i], std=stds[i]))
            f.write("\n")

    writer.flush()
    writer.close()


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    noise_rates = [0.3, 0.5, 0.7]

    datanames = ['emotions']
    # datanames = ['Pascal']
    # datanames = ['Corel5k']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']
    # datanames = ['Espgame']

    for dataname in datanames:
        for p in noise_rates:
            args.DATA_SET_NAME = dataname
            args.noise_rate = p
            save_dir = f'results/{dataname}/'
            save_name = f'{args.DATA_SET_NAME}-p{p}-r{args.noise_num}.txt'
            run(args, save_dir, save_name)

