import token
import torch
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import classification_report
import time
from datetime import datetime
from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from benchmarks_arguments import *
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import one_hot
import torch.nn as nn

#args = arguments(benchmark='glue', task='rte')

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss()
}


""" def parameters(model, args):
    encoder_params = {'params': [p for n, p in model.roberta.named_parameters() if p.requires_grad], 'weight_decay': args['weight_decay'], 'lr': args['lr']}
    classifier_params = {'params': [p for n, p in model.classifier.named_parameters() if p.requires_grad], 'weight_decay': args['weight_decay'], 'lr': args['lr']}
    return [encoder_params, classifier_params] """


def min_max_scale(tensor, min_val=-3, max_val=3):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # This scales tensor to [0, 1]
    return tensor * (max_val - min_val) + min_val  # This scales [0, 1] to [min_val, max_val]


class ScaledTanh(torch.nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.scale_factor * torch.tanh(x)


def z_score_norm(tensor, scale_factor = 1, epsilon=1e-10):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return scale_factor * (tensor - mean) / (std + epsilon)


def z_score_norm_2d(tensor, dim=0, scale_factor=1, epsilon=1e-10):
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    if torch.isnan(std).any():
        std = torch.where(torch.isnan(std), torch.ones_like(std), std)
    return scale_factor * (tensor - mean) / (std + epsilon)


class CustomNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-10):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdim=True)
        var = torch.var(x, dim=0, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)
        return  self.gamma * x_norm + self.beta


def parameters(model, args):
    params = {'params': [p for n, p in model.named_parameters() if p.requires_grad], 'weight_decay': args['weight_decay'], 'lr': args['lr']}
    return params


def galore_parameters(model):
    galore_params = []
    non_galore_params = []
    for name, param in model.named_parameters():

        if 'embeddings' in name and not 'LayerNorm' in name:
            galore_params.append(param)
            continue
        
        if 'layer' in name and 'weight' in name and not 'LayerNorm' in name:
            galore_params.append(param)
            continue

        if 'classifier' in name and not 'bias' in name:
            galore_params.append(param)
            continue
                      
        else:
            non_galore_params.append(param)
                
    param_groups = [{'params': non_galore_params},
                    {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}]   # 'proj_type': 'std', 'reverse_std','right', 'left', 'full'

    # initial rank is 128

    for param in galore_params:
        if param.dim() != 2:
            raise ValueError('Galore only supports 2D parameters')

    return param_groups
             


def calculate_warmup_steps(total_epochs, num_batches, warmup_proportion):
    total_steps = total_epochs * num_batches
    warmup_steps = int(total_steps * warmup_proportion)
    return warmup_steps


def select_scheduler(optimizer, lr_scheduler, num_epochs, num_batches, warmup_proportion):
    if lr_scheduler == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif lr_scheduler == 'warmup_constant':
            warmup_steps = calculate_warmup_steps(total_epochs=num_epochs, num_batches=num_batches, warmup_proportion=warmup_proportion)
            scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps= warmup_steps, num_training_steps=num_batches*num_epochs)
    elif lr_scheduler == 'warmup_linear':
        warmup_steps = calculate_warmup_steps(total_epochs=num_epochs, num_batches=num_batches, warmup_proportion=warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_batches*num_epochs)
    return scheduler

def reporting(y_true_list, y_pred_list, epoch, dict_name, dict={}, regression=False):
    if regression:
        mse = mean_squared_error(y_true_list, y_pred_list)
        r2 = r2_score(y_true_list, y_pred_list)
        pearson_corr, _ = pearsonr(y_true_list, y_pred_list)
        spearman_corr, _ = spearmanr(y_true_list, y_pred_list)
        print(f"{dict_name} REGRESSION REPORT:")
        print(f'Mean Squared Error: {mse}, R-Squared: {r2}, Pearson Correlation: {pearson_corr}, Spearman Correlation')
        dict = {'mse': mse, 'r2': r2, 'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr}
        with open(f"{dict_name}{epoch}.json", 'w') as fp:
            json.dump(dict, fp)
        return dict
    else:
        report = classification_report(y_true_list, y_pred_list, output_dict=True)
        print(f"{dict_name} CLASSIFICATION REPORT: ")
        print(classification_report(y_true_list, y_pred_list))
        dict = report
        with open(f"{dict_name}{epoch}.json", 'w') as fp:
            json.dump(dict, fp)
        return report


def train(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, dict_name, global_step, losses_dict_train, problem_type):
    # TRAINING
    log_interval = 10
    losses = []
    y_true_list = []
    y_pred_list = []
    model.train()
    start_time = time.time()
    # Usamos el train_loader para iterar sobre los datos de entrenamiento
    total_loss = 0 
    for i, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        try:
            input_ids = batch[0].to(device)
        except:
            input_ids = batch['input_ids'].to(device)
        try:
            attention_mask = batch[1].to(device)
        except:
            attention_mask = batch['attention_mask'].to(device)
        try:
            labels = batch[2]
        except:
            labels = batch['labels']
            labels = labels.float()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = outputs[0]
        outputs = outputs.squeeze(1)

        if problem_type == 'binary_classification' or problem_type == 'multi_class_classification':
            labels = labels.long()
            labels = one_hot(labels, num_classes=outputs.size(dim=1)).float()
        
        loss = loss_fn(outputs, labels)

        #loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Append the loss to the list of losses
        losses.append(loss.item())
        total_loss = loss.item()
        if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
            outputs = torch.sigmoid(outputs)
        y_true_list.extend(labels.detach().cpu().numpy())
        outputs = outputs.detach().cpu().numpy()
        try:
            outputs_results = np.argmax(outputs, axis=1)
        except:
            outputs_results = outputs
            outputs_results = (outputs_results > 0.5).astype(float)
        y_pred_list.extend(list(outputs_results))

        # Print info
        ms_per_batch = 1000 * (time.time() - start_time) / log_interval
        print('|step {:5} |lr: {:9.7f} |loss {:7.4f} |ms/batch {:7.2f}|'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
        
        start_time = time.time()
        global_step += 1
        losses_dict_train[epoch] = total_loss
        with open("losses_dict_train.json", 'w') as fp:
            json.dump(losses_dict_train, fp)

    # Report for each epoch
    _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name)

def train_benchmark(model, train_loader, loss_fn, optimizer, scheduler, device, epoch, dict_name, global_step, losses_dict_train):
    # TRAINING
    log_interval = 10
    losses = []
    y_true_list = []
    y_pred_list = []
    model.train()
    start_time = time.time()
    # Usamos el train_loader para iterar sobre los datos de entrenamiento
    total_loss = 0 
    for i, batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        try:
            labels = batch['labels']
            labels = labels.float()
        except:
            labels = batch['label']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = outputs[0]
        outputs = outputs.squeeze(1)
        loss = loss_fn(outputs, labels)

        #loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Append the loss to the list of losses
        losses.append(loss.item())
        total_loss = loss.item()
        y_true_list.extend(labels.detach().cpu().numpy())
        outputs = outputs.detach().cpu().numpy()
        if loss_fn != torch.nn.MSELoss():
            regression = False
            try:
                outputs_results = np.argmax(outputs, axis=1)
            except:
                outputs_results = outputs
        else:
            regression = True
            outputs_results = outputs
        y_pred_list.extend(list(outputs_results))

        # Print info
        ms_per_batch = 1000 * (time.time() - start_time) / log_interval
        print('|step {:5} |lr: {:9.7f} |loss {:7.4f} |ms/batch {:7.2f}|'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
        
        start_time = time.time()
        global_step += 1
        losses_dict_train[epoch] = total_loss
        with open("losses_dict_train.json", 'w') as fp:
            json.dump(losses_dict_train, fp)

    # Report for each epoch
    _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name, regression=regression)
    


def evaluation(model, dev_loader, loss_fn, device, epoch, dict_name, losses_dict_dev):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # Usamos el dev_loader para iterar sobre los datos de validación
        losses = []
        y_true_list = []
        y_pred_list = []
        total_loss = 0
        for batch in tqdm(dev_loader):
            # Get data and labels from batch
            try:
                input_ids = batch[0].to(device)
            except:
                input_ids = batch['input_ids'].to(device)
            try:
                attention_mask = batch[1].to(device)
            except:
                attention_mask = batch['attention_mask'].to(device)
            try:
                labels = batch[2]
            except:
                labels = batch['labels']
                labels = labels.float()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = outputs[0]
            outputs = outputs.squeeze(1)
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Classification Report
            if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
                outputs = torch.sigmoid(outputs)
            y_true_list.extend(labels.detach().cpu().numpy())
            outputs = outputs.detach().cpu().numpy()
            try:
                outputs_results = np.argmax(outputs, axis=1)
            except:
                outputs_results = outputs
                outputs_results = (outputs_results > 0.5).astype(float)
            y_pred_list.extend(list(outputs_results))
            losses.append(total_loss)
            losses_dict_dev[epoch] = losses
        with open("losses_dict_dev.json", 'w') as fp:
            json.dump(losses_dict_dev, fp)

        # Report and loss for each epoch
        _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name)
    
    return total_loss


def evaluation_benchmark(model, dev_loader, loss_fn, device, epoch, dict_name, losses_dict_dev):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # Usamos el dev_loader para iterar sobre los datos de validación
        losses = []
        y_true_list = []
        y_pred_list = []
        total_loss = 0
        for batch in tqdm(dev_loader):
            # Get data and labels from batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            try:
                labels = batch['labels']
                labels = labels.float()
            except:
                labels = batch['label']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = outputs[0]
            outputs = outputs.squeeze(1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Classification Report
            y_true_list.extend(labels.detach().cpu().numpy())
            outputs = outputs.detach().cpu().numpy()
            if loss_fn != torch.nn.MSELoss():
                regression = False
                try:
                    outputs_results = np.argmax(outputs, axis=1)
                except:
                    outputs_results = outputs
            else:
                regression = True
                outputs_results = outputs
            y_pred_list.extend(list(outputs_results))
            losses.append(total_loss)
            losses_dict_dev[epoch] = losses
        with open("losses_dict_dev.json", 'w') as fp:
            json.dump(losses_dict_dev, fp)

        # Report and loss for each epoch
        _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name, regression=regression)
    
    return total_loss


def test(model, test_loader, device, epoch, dict_name):
            # TESTING
        model.eval()
        # Usamos el test_loader para iterar sobre los datos de test
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Get data and labels from batch
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                try:
                    labels = batch[2]
                except:
                    labels = batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs[0]
                # Classification Report
                y_true_list.extend(labels.detach().cpu().numpy())
                outputs = outputs.detach().cpu().numpy()
                outputs_results = np.argmax(outputs, axis=1)
                y_pred_list.extend(list(outputs_results))

        # Report and loss for each epoch
        _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name)


def test_benchmark(model, test_loader, device, epoch, dict_name):
            # TESTING
        model.eval()
        # Usamos el test_loader para iterar sobre los datos de test
        y_true_list = []
        y_pred_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Get data and labels from batch
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                try:
                    labels = batch['labels']
                except:
                    labels = batch['label']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                outputs = outputs[0]
                # Classification Report
                y_true_list.extend(labels.detach().cpu().numpy())
                outputs = outputs.detach().cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                y_pred_list.extend(list(outputs))

        # Report and loss for each epoch
        _ = reporting(y_true_list, y_pred_list, epoch=epoch, dict_name=dict_name)
        
        
def cast_labels_to_float(example):
    example['labels'] = torch.tensor(example['labels'], dtype=torch.float).tolist()
    return example