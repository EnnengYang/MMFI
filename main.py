import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np
import random

from datasets.aliexpress_all import AllDataset_DT
from datasets.aliexpress_single import AliExpressDataset
from models.mmfi import MMFIModel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_dataset(name, path, scenario_id):
    if 'AliExpress' in name:
        return AliExpressDataset(path, scenario_id)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_all_dataset_dt(name, dataset_paths, jay_paths, divided_paths):
    if name == 'ALL':
        return AllDataset_DT(dataset_paths, jay_paths, divided_paths)
    else:
        raise ValueError('unknown dataset name: ' + name)

class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if torch.distributed.get_rank() == 0:
                torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

class EarlyStoppers(object):

    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = [0, 0]
        self.loss = [0, 0]

    def is_continuable(self, accuracy, loss):
        if np.array(accuracy).mean() > np.array(self.best_accuracy).mean():
            self.best_accuracy = accuracy
            self.loss = loss
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)
        y = model(categorical_fields, numerical_fields, device)
        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)
            y = model(categorical_fields, numerical_fields, device)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(
                    torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results

def main(dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir,
        ):
    scenarios = ['NL', 'ES', 'FR', 'US']
    device = torch.device(device)

    if dataset_name == 'ALL':
        train_dataset_paths = []
        test_dataset_paths = []
        test_data_loaders = []
        train_jay_paths = []
        test_jay_paths = []

        for i in range(4):
            train_dataset_paths.append(os.path.join('./data/', 'AliExpress_' + scenarios[i]) + '/train.csv')
            test_dataset_paths.append(os.path.join('./data/', 'AliExpress_' + scenarios[i]) + '/test.csv')
            train_jay_paths.append(os.path.join('./datasets/', 'AliExpress_' + scenarios[i]) + '_train.jay')
            test_jay_paths.append(os.path.join('./datasets/', 'AliExpress_' + scenarios[i]) + '_test.jay')

        train_divided_paths = ['./datasets/train_categorical.jay', './datasets/train_numerical.jay',
                               './datasets/train_labels.jay']
        test_divided_paths = ['./datasets/test_categorical.jay', './datasets/test_numerical.jay',
                              './datasets/test_labels.jay']

        print('Reading all train datasets')
        train_dataset = get_all_dataset_dt(dataset_name, train_dataset_paths, train_jay_paths, train_divided_paths)
        print('Reading all test datasets')
        test_all_dataset = get_all_dataset_dt(dataset_name, test_dataset_paths, test_jay_paths, test_divided_paths)
        test_all_data_loader = DataLoader(test_all_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
        del test_all_dataset

        for i in range(4):
            print('Reading {} test dataset'.format(scenarios[i]))
            test_dataset = get_dataset('AliExpress_' + scenarios[i],
                                       os.path.join('./data/', 'AliExpress_' + scenarios[i]) + '/test.csv', i)
            test_data_loaders.append(DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False))
        del test_dataset
    else:
        id = 0
        if dataset_name == 'AliExpress_NL':
            id = 0
        elif dataset_name == 'AliExpress_ES':
            id = 1
        elif dataset_name == 'AliExpress_FR':
            id = 2
        elif dataset_name == 'AliExpress_US':
            id = 3
        else:
            raise ValueError('unknown model name: ' + dataset_name)
        train_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv', id)
        test_dataset = get_dataset(dataset_name, os.path.join(dataset_path, dataset_name) + '/test.csv', id)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    print('mmfi')
    print(field_dims)

    random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.backends.cudnn.deterministic = True

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    model = MMFIModel(task_num, embed_dim, numerical_num, field_dims, expert_num).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    save_path = f'{save_dir}/{dataset_name}_{model_name}.pt'

    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    early_stoppers = {}
    for s in scenarios:
        early_stoppers[s] = EarlyStoppers(num_trials=5)

    if dataset_name == 'ALL':
        for epoch_i in range(0, epoch):
            train(model, optimizer, train_data_loader, criterion, device)

            f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
            print('epoch:', epoch_i)
            f.write('epoch:' + str(epoch_i) + '   ')
            f.write('learning rate: {}\n'.format(learning_rate))

            id = 0
            while id < len(scenarios):
                auc, loss = test(model, test_data_loaders[id], task_num, device)
                f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')

                print(scenarios[id], ':')
                f.write(scenarios[id] + ': \n')
                print('mean auc:', np.array(auc).mean())
                f.write('mean auc: ' + str(np.array(auc).mean()) + '\n')
                for i in range(task_num):
                    print(
                        'task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
                    f.write(
                        'task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
                if not early_stoppers[scenarios[id]].is_continuable(auc, loss):
                    scenarios.pop(id)
                    test_data_loaders.pop(id)
                    id -= 1
                id += 1

            print('ALL:')
            f.write('ALL:\n')
            auc, loss = test(model, test_all_data_loader, task_num, device)
            print('mean auc:', np.array(auc).mean())
            f.write('mean auc: ' + str(np.array(auc).mean()) + '\n')
            for i in range(task_num):
                print(
                    'task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
                f.write(
                    'task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
            if not early_stopper.is_continuable(model, np.array(auc).mean()):
                print(f'test: best auc: {early_stopper.best_accuracy}')
                f.write(f'test: best auc: {early_stopper.best_accuracy}')
                break
            f.close()

        scenarios = ['NL', 'ES', 'FR', 'US']
        f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
        f.write('learning rate: {}\n'.format(learning_rate))

        for id in range(4):
            print(scenarios[id] + ':')
            f.write(scenarios[id] + ':\n')
            auc = early_stoppers[scenarios[id]].best_accuracy
            loss = early_stoppers[scenarios[id]].loss
            for i in range(task_num):
                print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
                f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))

        model.load_state_dict(torch.load(save_path))
        auc, loss = test(model, test_all_data_loader, task_num, device)

        print('ALL:')
        f.write('ALL:\n')
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
            f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
        print('\n')
        f.write('\n\n')
        f.close()
    else:
        for epoch_i in range(epoch):
            train(model, optimizer, train_data_loader, criterion, device)
            auc, loss = test(model, test_data_loader, task_num, device)
            f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
            f.write('learning rate: {}\n'.format(learning_rate))
            print('epoch:', epoch_i, 'test: auc:', auc)
            for i in range(task_num):
                print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
                f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
            print("\n")
            if not early_stopper.is_continuable(model, np.array(auc).mean()):
                print(f'test: best auc: {early_stopper.best_accuracy}')
                f.write(f'test: best auc: {early_stopper.best_accuracy}\n')
                break
            f.close()

        model.load_state_dict(torch.load(save_path))
        auc, loss = test(model, test_data_loader, task_num, device)
        f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding='utf-8')
        f.write('learning rate: {}\n'.format(learning_rate))
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
            f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
        print('\n')
        f.write('\n\n')
        f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'ALL'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='mmfi')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./chkpt')
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         )