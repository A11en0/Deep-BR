# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1

class Args:
    def __init__(self):
        self.DATA_ROOT = './datasets'
        self.DATA_SET_NAME = 'Espgame'
        self.epoch = 10
        self.show_epoch = 1
        # self.epoch_used_for_final_result = 4
        self.model_save_epoch = 10
        self.model_save_dir = 'model_save_dir'

        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 8
        self.cuda = True
        self.opt = 'adam'
        self.lr = 1e-3  # 1e-3 5e-3
        self.weight_decay = 1e-5  # 1e-5
        self.noise_rate = 0.7
        self.noise_num = 3

        self.feature_dim = 2048
        self.keep_prob = 0.5
        self.scale_coeff = 1.0


loss_coefficient = {}

