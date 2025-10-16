import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import os
import h5py
import numpy as np
import random
import copy
from eegnet import EEGNet
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix
import math
import pickle
import logging

from datetime import datetime


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
g = torch.Generator()
g.manual_seed(42)

currenttime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.getcwd()), f'training_thielen_{currenttime}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
 )

logging.info('Starting.')
 
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'])
        y = np.array(f['y'])
    return X, y
 
def load_data(file_list):
    X_list = []
    y_list = []
    for file in file_list:
        X, y = read_hdf5(file)
        X_list.append(X)
        y_list.append(y)

    X_combined = np.concatenate(X_list, axis=0)
    y_combined = np.concatenate(y_list, axis=0)
    return X_combined, y_combined

class EarlyStopper:
    def __init__(self, patience=40, min_delta=0, smoothing_window=1):

        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.counter = 0
        self.min_loss = float('inf')
        self.loss_history = []

    def early_stop(self, current_loss):

        self.loss_history.append(current_loss)
        smoothed_loss = np.mean(self.loss_history[-self.smoothing_window:]) if len(self.loss_history) >= self.smoothing_window else current_loss
        
        if smoothed_loss < (self.min_loss - self.min_delta):
            self.min_loss = smoothed_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
                
        return False

    def reset(self):

        self.counter = 0
        self.min_loss = float('inf')
        self.loss_history = []

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                        self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr
    

datasets = ['thielen2021']

conditions = ['vanilla','vanilla_nonorm_decay', 'mixup',  'layernorm', 'sae','ema']


condition_dict = {condition: False for condition in conditions}

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

criterion = nn.CrossEntropyLoss()

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def update_conditions(iteration):
    for i, condition in enumerate(conditions):
        if i < iteration:
            condition_dict[condition] = True
        else:
            condition_dict[condition] = False
            
def update_conditions(iteration):
    for i, condition in enumerate(conditions):
        if i == iteration:
            condition_dict[condition] = True
        else:
            condition_dict[condition] = False
            
def max_norm_(w, max_norm_val, dim=1, eps=1e-7):
    with torch.no_grad():
        norm = w.norm(2, dim=dim, keepdim=True)
        norm = norm.clamp(min=max_norm_val/2)
        desired= torch.clamp(norm, max=max_norm_val)
        w *= (desired / (norm + eps))
        
def check_norms(model, layer_names, max_val=1.0, dim=1):
      """Check if weights exceed max norm constraint"""
      with torch.no_grad():
            for name, param in model.named_parameters():
                  if any(layer in name for layer in layer_names):
                        norms = param.norm(2, dim=dim)   # L2 norm along specified dim
                        max_observed = norms.max().item()
                        logging.info(f"{name}: max_norm={max_observed:.5f} (allowed ≤ {max_val})")


test_conditions = [['vanilla'], ['nonorm_decay'], ['nonorm_decay', 'mixup'], ['nonorm_decay', 'sae'], ['nonorm_decay', 'layernorm'],['nonorm_decay', 'ema'], ['nonorm_decay', 'layernorm', 'mixup'], ['nonorm_decay', 'layernorm', 'sae'], ['nonorm_decay', 'layernorm', 'ema'],['nonorm_decay', 'layernorm', 'ema', 'sae'], ['nonorm_decay', 'layernorm', 'ema', 'mixup'], ['nonorm_decay', 'layernorm', 'mixup', 'sae'], ['nonorm_decay', 'layernorm', 'mixup', 'sae', 'ema']]

results_dict = {
    'thielen2021': {tuple(cond): [] for cond in test_conditions},
}


n_iters = 30
num_epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for dataset in datasets:
    currenttime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
    subject_list = glob.glob(os.path.join(os.path.dirname(os.getcwd()), 'datasets/*') + dataset + "/*")
    subject_list = [file for file in subject_list if 'multicycle' not in file][:12]
    logging.info(subject_list)
    train_id, val_id = train_test_split(subject_list, test_size=0.30, random_state=42)
    val_id, test_id = train_test_split(val_id, test_size=0.50, random_state=42)

    
    X_train, y_train = load_data(train_id)
    X_val, y_val = load_data(val_id)
    X_test, y_test = load_data(test_id)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    
    class_counts = torch.bincount(y_train)
    minority_class = torch.argmin(class_counts)
    class_weights = torch.ones_like(class_counts).float().to(device)
    class_weights.requires_grad = False

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)

    batch_size = 32

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    n_classes = len(torch.unique(y_train))
    
    for test_condition in test_conditions:
        condition_dict = {'mixup': False, 'vanilla': False, 'sae': False, 'layernorm': False, 'ema': False, 'ema999': False, 'nonorm_decay': False}
        for subcondition in test_condition:
            condition_dict[subcondition]= True
        logging.info(f"Current condition {condition_dict}")
        
        for iter in range(n_iters):
            #set_seed(42)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True, generator=g,worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, persistent_workers=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, persistent_workers=False)
            
            logging.info(f"Iteration {iter + 1}/{n_iters}")

            model = EEGNet(sampling_fq=(100 if dataset=='lee2019' else 128), data_shape=(n_channels, n_times), n_classes=n_classes, dropout=0.5, layernorm=condition_dict['layernorm'], sae=condition_dict['sae']).to(device)   
            logging.info(model)
            if condition_dict['ema']: 
                ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995)).to(device) 
                
            criterion = nn.CrossEntropyLoss(weight=class_weights) #
            if condition_dict['nonorm_decay']:
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            early_stopper = EarlyStopper()   


            best_val_loss = float("inf")

            for epoch in range(num_epochs):

                model.train()
                if condition_dict['ema']: 
                    ema_model.train()
                
                running_loss = 0.0
                
                for inputs, labels in train_loader:


                    inputs, labels = inputs.to(device), labels.to(device)
                    if condition_dict['mixup']:    
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels,
                                                            1., torch.cuda.is_available())
                        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                                    targets_a, targets_b))
                        
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    if condition_dict['mixup']:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    
                    optimizer.step()
                    running_loss += loss.item()

                    if condition_dict['vanilla']:
                        with torch.no_grad():
                            max_norm_(model.depthwiseconv1.weight, 1.0, dim=(2,3))
                            max_norm_(model.linear.weight, 0.25, dim=1)
      
                    if condition_dict['ema']:
                       ema_model.update_parameters(model)
                    
                
                model.eval()
                if condition_dict['ema']: 
                    ema_model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0
                val_all_labels = []
                val_all_predictions = []
                if condition_dict['ema']:
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                            val_outputs = ema_model(val_inputs)
                            val_loss = criterion(val_outputs, val_labels)
                            val_running_loss += val_loss.item()
                            _, val_predicted = torch.max(val_outputs.data, 1)
                            val_total += val_labels.size(0)
                            val_correct += (val_predicted == val_labels).sum().item()
                            val_all_labels.extend(val_labels.cpu().numpy())
                            val_all_predictions.extend(val_predicted.cpu().numpy())
                else:
                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                            val_outputs = model(val_inputs)
                            val_loss = criterion(val_outputs, val_labels)
                            val_running_loss += val_loss.item()
                            _, val_predicted = torch.max(val_outputs.data, 1)
                            val_total += val_labels.size(0)
                            val_correct += (val_predicted == val_labels).sum().item()
                            val_all_labels.extend(val_labels.cpu().numpy())
                            val_all_predictions.extend(val_predicted.cpu().numpy())
                
                val_accuracy = 100 * val_correct / val_total
                val_loss = val_running_loss/len(val_loader)
                if (epoch+1) % 10==0:
                    logging.info(f"Epoch {epoch+1}, val loss: {val_loss}, val accuracy: {val_accuracy}%")
                
                    # Early stopping check
                if early_stopper is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                        if condition_dict['ema']:
                            best_ema_weights = ema_model.state_dict()
                        else:
                            best_weights = model.state_dict()
                            
                        
                if early_stopper.early_stop(val_loss):
                    logging.info(f"Early stopping triggered at {best_val_loss}")
                    break

            if condition_dict['ema']:
                ema_model.load_state_dict(best_ema_weights)
                if not condition_dict['layernorm']:
                    torch.optim.swa_utils.update_bn(train_loader, ema_model, device=device)
                model = copy.deepcopy(ema_model)
            else:
                model.load_state_dict(best_weights)
            
                
            model.eval()
            
            test_running_loss = 0.0
            test_correct = 0
            test_total = 0
            test_all_labels = []
            test_all_predictions = []
            with torch.no_grad():
                for test_inputs, test_labels in test_loader:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_labels)
                    test_running_loss += test_loss.item()
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_total += test_labels.size(0)
                    test_correct += (test_predicted == test_labels).sum().item()
                    test_all_labels.extend(test_labels.cpu().numpy())
                    test_all_predictions.extend(test_predicted.cpu().numpy())

            test_accuracy = 100 * test_correct / test_total
            test_loss = test_running_loss/len(test_loader)
            conf_matrix = confusion_matrix(test_all_labels, test_all_predictions)
            logging.info(f"Training end, Test loss: {test_loss}, Test accuracy: {test_accuracy}%")
            logging.info("Test confusion Matrix:")
            logging.info(conf_matrix)
            results_dict[dataset][tuple(test_condition)].append((test_accuracy, conf_matrix))
    with open(f"results_{dataset}_thielen_{currenttime}.pkl", 'wb') as f:
        pickle.dump(results_dict, f)

logging.info(results_dict)

