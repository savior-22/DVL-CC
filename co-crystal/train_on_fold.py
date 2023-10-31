from datetime import datetime
import numpy as np
import torch
from torch import optim
import models
import time
from tqdm import tqdm
from ddi_datasets import load_ddi_data_fold, total_num_rel
from custom_loss import SigmoidLoss
from custom_metrics import do_compute_metrics
import argparse
import dill as pickle
import warnings


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank', 'twosides'])
parser.add_argument('-fold', '--fold', type=int, required=True, help='Fold on which to train on')
parser.add_argument('-n_iter', '--n_iter', type=int, required=True, help='Number of iterations/')
parser.add_argument('-drop', '--dropout', type=float, default=0, help='dropout probability')
parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size')

args = parser.parse_args()

print(args)

dataset_name = args.dataset
fold_i = args.fold
dropout = args.dropout
n_iter = args.n_iter
TOTAL_NUM_RELS = total_num_rel(name=dataset_name)
batch_size = args.batch_size
data_size_ratio = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hid_feats = 64
rel_total = TOTAL_NUM_RELS
lr = 1e-3 #1,3
weight_decay = 5e-4
n_epochs = 100
kge_feats = 64
eps = 1e-6
LR_G = 0.0001
LR_D = 0.0001
def do_compute(model,modelG1, modelG2, modelD, batch, device): 

        batch = [t.to(device) for t in batch]
        p_score, n_score, view1_specific , view2_specific , view1_shared , view2_shared,d1 ,d2 , tx, txsm = model(modelG1, modelG2, modelD,batch)


        assert p_score.ndim == 2
        assert n_score.ndim == 3
        probas_pred = np.concatenate([torch.sigmoid(p_score.detach()).cpu().mean(dim=-1), torch.sigmoid(n_score.detach()).mean(dim=-1).view(-1).cpu()])
        ground_truth = np.concatenate([np.ones(p_score.shape[0]), np.zeros(n_score.shape[:2]).reshape(-1)])
        return p_score, n_score, probas_pred, ground_truth, view1_specific , view2_specific , view1_shared , view2_shared,d1,d2 ,tx, txsm


def run_batch(model,modelG1, modelG2, modelD,  optimizer, data_loader, epoch_i, desc, loss_fn, device, optimizer_G1, optimizer_G2,  optimizer_D1):
        total_loss = 0
        loss_pos = 0
        loss_neg = 0
        probas_pred = []
        ground_truth = []
        torch.autograd.set_detect_anomaly(True)
        for batch in tqdm(data_loader, desc= f'{desc} Epoch {epoch_i}'):
            p_score, n_score, batch_probas_pred, batch_ground_truth, view1_specific , view2_specific , view1_shared , view2_shared ,d1 ,d2 ,tx ,txsm= do_compute(model,modelG1, modelG2, modelD, batch, device)

            probas_pred.append(batch_probas_pred)
            ground_truth.append(batch_ground_truth)

            loss, loss_p, loss_n = loss_fn(p_score, n_score, view1_specific , view2_specific , view1_shared , view2_shared)
            G1_loss = -1/torch.mean(torch.log(eps+1.-d1))

            G2_loss = -1/torch.mean(torch.log(eps+1.-d2))

            D1_loss = -torch.mean(torch.log(eps +tx)+torch.log(eps +1.-d1))-torch.mean(torch.log(eps +txsm) + torch.log(eps +1. - d2))


            if model.training:
                optimizer.zero_grad()
                optimizer_G1.zero_grad()
                optimizer_G2.zero_grad()
                optimizer_D1.zero_grad()
                #optimizer_D2.zero_grad()

                D1_loss.backward(retain_graph=True)
                #D2_loss.backward(retain_graph=True)
                G1_loss.backward()
                G2_loss.backward()
                loss.backward()

                optimizer_D1.step()
                #optimizer_D2.step()
                optimizer_G1.step()
                optimizer_G2.step()
                optimizer.step()



            total_loss += loss.item()
            #loss_pos += loss_p.item()
            #loss_neg += loss_n.item()
        total_loss /= len(data_loader)
        #loss_pos /= len(data_loader)
        #loss_neg /= len(data_loader)
        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)
        print(probas_pred)

        #print(probas_pred)
        #print(ground_truth)
        np.savetxt("test.csv", probas_pred)
        return total_loss, do_compute_metrics(probas_pred, ground_truth)


def print_metrics(loss, acc, auroc, f1_score, precision, recall, int_ap, ap, pred,target):
    print(f'loss: {loss:.4f}, acc: {acc:.4f}, roc: {auroc:.4f}, f1: {f1_score:.4f}, ', end='')
    print(f'p: {precision:.4f}, r: {recall:.4f}, int-ap: {int_ap:.4f}, ap: {ap:.4f}')  
    #print(pred)

    return f1_score,pred,target



def train(model, modelG1, modelG2, modelD, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs
          , device, scheduler):
    acc_array=[]
    loss_array=[]
    auc_array=[]
    time11 = []

    for epoch_i in range(1, n_epochs+1):
        start = time.time()
        model.train()
        modelG1.train()
        modelG2.train()
        modelD.train()
        ## Training
        train_loss, train_metrics = run_batch(model, modelG1, modelG2, modelD, optimizer, train_data_loader, epoch_i,  'train', loss_fn, device,optimizer_G1, optimizer_G2,  optimizer_D1)
        if scheduler:
            scheduler.step()

        model.eval()
        modelG1.eval()
        modelG2.eval()
        modelD.eval()
        with torch.no_grad():

            ## Validation 
            if val_data_loader:
                val_loss , val_metrics = run_batch(model, modelG1, modelG2, modelD, optimizer, val_data_loader, epoch_i, 'val', loss_fn, device,optimizer_G1, optimizer_G2,  optimizer_D1)
            #if epoch_i==100:
            if test_data_loader:
                test_loss, test_metrics = run_batch(model, modelG1, modelG2, modelD, optimizer, test_data_loader, epoch_i, 'test', loss_fn,
                                                    device,optimizer_G1, optimizer_G2, optimizer_D1)

        if epoch_i==100:
            with open(r'fold'+str(fold_i)+'.model', 'wb') as file:
                pickle.dump(model,file)
            with open(r'fold'+str(fold_i)+'modelG1.model', 'wb') as file:
                pickle.dump(modelG1,file)
            with open(r'fold'+str(fold_i)+'modelG2.model', 'wb') as file:
                pickle.dump(modelG2,file)
            with open(r'fold'+str(fold_i)+'modelD.model', 'wb') as file:
                pickle.dump(modelD,file)



        if train_data_loader:
            print(f'\n#### Epoch time {time.time() - start:.4f}s')
            time11.append(time.time() - start)
            print_metrics(train_loss, *train_metrics)
            acc_array.append(train_metrics[0])
            loss_array.append(train_loss)
            auc_array.append(train_metrics[5])



        if val_data_loader:
            print('#### Validation')
            print_metrics(val_loss, *val_metrics)
            #print_metrics(val_loss11, *val_metrics11)

        #if epoch_i==100:
        if test_data_loader:
            print('#### Test')
            _,pred,target=print_metrics(test_loss, *test_metrics)
            if epoch_i==100:
                np.savetxt("traget.csv", target)
                #np.savetxt("test.csv", pred)

        #print_metrics(test_loss11, *test_metrics11)

    '''with open(r'fold'+str(fold_i)+'.model', 'rb') as file:
        the_model = pickle.load(file)
    with open(r'fold'+str(fold_i)+'modelG1.model', 'rb') as file:
        modelG1 = pickle.load(file)
    with open(r'fold'+str(fold_i)+'modelG2.model', 'rb') as file:
        modelG2 = pickle.load(file)
    with open(r'fold'+str(fold_i)+'modelD.model', 'rb') as file:
        modelD = pickle.load(file)
    the_model.to(device=device)
    modelG1.to(device=device)
    modelG2.to(device=device)
    modelD.to(device=device)
    test_loss1, test_metrics1 = run_batch(the_model, modelG1, modelG2, modelD, optimizer, test_data_loader, 1, 'test', loss_fn,
                                          device, optimizer_G1, optimizer_G2, optimizer_D1)
    print_metrics(test_loss1, *test_metrics1)'''





train_data_loader, val_data_loader, test_data_loader, NUM_FEATURES, NUM_EDGE_FEATURES = \
    load_ddi_data_fold(dataset_name, fold_i, batch_size=batch_size, data_size_ratio=data_size_ratio)

GmpnnNet = models.GmpnnCSNetDrugBank if dataset_name == 'drugbank' else models.GmpnnCSNetTwosides

model = GmpnnNet(NUM_FEATURES, NUM_EDGE_FEATURES, hid_feats, rel_total, n_iter, dropout)
modelG1 = models.gen1
modelG2 = models.gen2
modelD = models.dis1
loss_fn = SigmoidLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
optimizer_G1 = optim.Adam(models.gen1.parameters(), lr=LR_G)
optimizer_G2 = optim.Adam(models.gen2.parameters(), lr=LR_G)
optimizer_D1 = optim.Adam(models.dis1.parameters(), lr=LR_D)


time_stamp = f'{datetime.now()}'.replace(':', '_')


model.to(device=device)
print(f'Training on {device}.')
print(f'Starting fold_{fold_i} at', datetime.now())
train(model,modelG1, modelG2, modelD, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, n_epochs, device, scheduler)
