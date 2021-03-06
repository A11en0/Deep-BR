# -*- coding: UTF-8 -*-
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.ml_metrics import all_metrics


def train(model, device, views_data_loader, args, loss_coefficient,
          train_features, train_partial_labels, test_features, test_labels, fold=1):

    # init optimizer
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentumae)

    # train model
    trainer = Trainer(model, views_data_loader, args.epoch, optimizer, args.show_epoch,
                      loss_coefficient, args.model_save_epoch, args.model_save_dir, device)
    loss_list = trainer.fit(fold, train_features, train_partial_labels, test_features, test_labels, args)

    return loss_list

@ torch.no_grad()
def test(model, features, labels, device, model_state_path=None, is_eval=False, args=None):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))

    metrics_results = None

    model.eval()

    # CUDA
    for i, _ in enumerate(features):
        features[i] = features[i].to(device)
    labels = labels.to(device)

    # prediction
    with torch.no_grad():
        outputs = model(features)

    outputs = outputs.cpu().numpy()
    preds = (outputs > 0.5).astype(int)

    # eval
    if is_eval:
        target = labels.int().cpu().numpy()
        metrics_results = all_metrics(outputs, preds, target)

    return metrics_results, preds

class Trainer(object):
    def __init__(self, model, train_data_loader, epoch, optimizer, show_epoch,
                 loss_coefficient, model_save_epoch, model_save_dir, device):
        self.model = model
        self.train_data_loader = train_data_loader
        self.epoch = epoch
        self.optimizer = optimizer
        self.show_epoch = show_epoch
        self.loss_coefficient = loss_coefficient
        self.model_save_epoch = model_save_epoch
        self.model_save_dir = model_save_dir
        self.device = device

    def fit(self, fold, train_features, train_partial_labels, test_features, test_labels, args=None):
        loss_list = []
        best_F1 = 0.0
        best_epoch = 0
        writer = SummaryWriter()
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        for epoch in range(self.epoch):
            self.model.train()
            for step, train_data in enumerate(self.train_data_loader):
                inputs, labels, index = train_data

                # CUDA Pay attention! Do Not migrate data to CUDA in Dataset class!
                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, labels)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}\t'

                # show loss info
                if epoch % self.show_epoch == 0 and step == 0:
                    epoch_loss = dict()
                    writer.add_scalar("Loss/train", loss, epoch)  # log
                    # plotter.plot('loss', 'train', 'Class Loss', epoch, _ML_loss)
                    loss_list.append(epoch_loss)
                    print(print_str)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # evaluation
            if epoch % self.show_epoch == 0 and args.is_test_in_train:
                metrics_results, _ = test(self.model, test_features, test_labels,
                                                  self.device, is_eval=True, args=args)

                # draw figure to find best epoch number
                loss_list[epoch]["Hamming"] = metrics_results[0][1]
                loss_list[epoch]["Average"] = metrics_results[1][1]
                loss_list[epoch]["OneError"] = metrics_results[2][1]
                loss_list[epoch]["Ranking"] = metrics_results[3][1]
                loss_list[epoch]["Coverage"] = metrics_results[4][1]
                loss_list[epoch]["MacroF1"] = metrics_results[5][1]
                loss_list[epoch]["MicroF1"] = metrics_results[6][1]

                if best_F1 < metrics_results[6][1]:
                    best_F1, best_epoch = metrics_results[6][1], epoch

                metrics = ['hamming_loss', 'avg_precision', 'one_error', 'ranking_loss', 'coverage', 'macrof1',
                           'microf1']
                for i in range(7):
                    print(f"{metrics[i]}: {metrics_results[i][1]:.4f}", end='\t')
                print("\n")

            if (epoch + 1) % self.model_save_epoch == 0:
                torch.save(self.model.state_dict(),
                        os.path.join(self.model_save_dir,
                                     'fold' + str(fold)+'_' + 'epoch' + str(epoch + 1) + '.pth'))
        writer.flush()
        writer.close()
        print(f"best_F1: {best_F1}, epoch {best_epoch}")
        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)