import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from torch_scatter import scatter
from layers import (CoAttentionLayerDrugBank,
                    CoAttentionLayerTwosides,
                    )
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from spect_conv import SpectConv,ML3Layer
import torch.nn.functional as F
from torch_geometric.nn import (global_max_pool,global_mean_pool,global_add_pool)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.manifold import TSNE

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',200)
def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功")


class GmpnnCSNetDrugBank(nn.Module):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):
        super().__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.rel_total = rel_total
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_hid_feats = hid_feats * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(hid_feats, hid_feats),
            nn.BatchNorm1d(hid_feats), 
            CustomDropout(self.dropout),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(94, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
        )


        #self.gcn_layer = GCNBlock(in_feats, self.hid_feats, 128) //gcn
        self.gcn_layer = GCNBlock(in_feats, self.hid_feats, 64) #//prechuli
        self.gcn = GCN(in_feats, self.hid_feats, self.snd_hid_feats)
        self.gat = GAT(in_feats, self.hid_feats, self.snd_hid_feats)
        #self.gcn_layer=GCNBlock(self.hid_feats,self.hid_feats,self.snd_hid_feats)
        self.GNNML=GNNML1(64,128)
        self.GNNML1 = GNNML1(128, 128)
        self.GNNML2 = GNNML1(128, 128)
        self.GNNML3 = GNNML1(128, 128)
        self.GNNML4 = GNNML1(128, 128)

        self.propagation_layer = GmpnnBlock(edge_feats, 128, self.n_iter, dropout)
        self.Mpnn = Mpnn(64,128)
        self.Mpnn1 = Mpnn(128, 128)
        self.Mpnn2 = Mpnn(128, 128)
        self.Mpnn3 = Mpnn(128, 128)
        self.Mpnn4 = Mpnn(128, 128)
        self.Mpnn5 = Mpnn(128, 128)
        self.Mpnn6 = Mpnn(128, 128)
        self.Mpnn7 = Mpnn(128, 128)
        self.Mpnn8 = Mpnn(128, 128)
        self.bn = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        #self.propagation_layer = GmpnnBlock(edge_feats, self.hid_feats, self.n_iter, dropout)
        self.i_pro = nn.Parameter(torch.zeros(self.snd_hid_feats , self.hid_feats))
        self.j_pro = nn.Parameter(torch.zeros(self.snd_hid_feats, self.hid_feats))
        self.bias = nn.Parameter(torch.zeros(self.hid_feats ))
        self.co_attention_layer = CoAttentionLayerDrugBank(self.snd_hid_feats)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats)


        glorot(self.i_pro)
        glorot(self.j_pro)


        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_hid_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_hid_feats, self.snd_hid_feats)
        )

    def forward(self, batch):
        drug_data, unique_drug_pair, rels, drug_pair_indices, node_j_for_pairs, node_i_for_pairs = batch
        drug_data.x = self.mlp(drug_data.x)
        new_feats = self.propagation_layer(drug_data)
        drug_data.x = new_feats
        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        attentions = self.co_attention_layer(x_j, x_i, unique_drug_pair)
        pair_repr = attentions.unsqueeze(-1) * ((x_i[unique_drug_pair.edge_index[1]] @ self.i_pro) * (x_j[unique_drug_pair.edge_index[0]] @ self.j_pro))
        
        x_i = x_j = None ## Just to free up some memory space
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None

        pair_repr = scatter(pair_repr, unique_drug_pair.edge_index_batch, reduce='add', dim=0)[drug_pair_indices]
        p_scores, n_scores = self.compute_score(pair_repr, rels)
        return p_scores, n_scores
    
    def compute_score(self, pair_repr, rels):
        batch_size = len(rels)
        neg_n = (len(pair_repr) - batch_size) // batch_size  # I case of multiple negative samples per positive sample.
        rels = torch.cat([rels, torch.repeat_interleave(rels, neg_n, dim=0)], dim=0)
        rels = self.rel_embs(rels)
        scores = (pair_repr * rels).sum(-1)
        p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)

        return p_scores, n_scores


class GmpnnCSNetTwosides(GmpnnCSNetDrugBank):
    def __init__(self, in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout=0):
        super().__init__(in_feats, edge_feats, hid_feats, rel_total, n_iter, dropout)
        self.co_attention_layer = CoAttentionLayerDrugBank(self.hid_feats * 2)
        self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats)
        #self.rel_embs = nn.Embedding(self.rel_total, self.hid_feats * 2)
        self.rel_proj = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats * 2),
            nn.PReLU(),
            nn.Linear(self.hid_feats * 2, self.hid_feats),
        )
        self.s_pro = self.i_pro
        self.j_pro = self.i_pro =  None
        self.fclayer1 = nn.Linear(192,256)
        self.fclayer2 = nn.Linear(384, 192)
        self.fclayer3 = nn.Linear(256, 128)
        self.fclayer = nn.Linear(128,1)
        self.m = nn.Softmax(dim=1)
        self.ts =TSNE(n_components=2, init = 'pca',random_state=0)
        self.embs = nn.Embedding(117, 128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        # View1
        self.shared1_l1 = nn.Linear(32, 64)
        self.shared1_l2 = nn.Linear(64, 64)
        self.shared1_l3 = nn.Linear(64, 32)

        self.specific1_l1 = nn.Linear(32, 64)
        self.specific1_l2 = nn.Linear(64, 64)
        self.specific1_l3 = nn.Linear(64, 32)

        # View2
        self.shared2_l1 = nn.Linear(32, 64)
        self.shared2_l2 = nn.Linear(64, 64)
        self.shared2_l3 = nn.Linear(64, 32)

        self.specific2_l1 = nn.Linear(32, 64)
        self.specific2_l2 = nn.Linear(64, 64)
        self.specific2_l3 = nn.Linear(64, 32)
        self.gen1 = gen1()
        self.gen2 = gen2()
        self.dis1 = dis1()
        self.dis2 = dis2()



    def forward(self, batch):
        drug_data, drug_pairs, rels, batch_size, node_j_for_pairs, node_i_for_pairs = batch
        rels1 = torch.zeros(batch_size)
        rels1 = rels1.long()
        rels1 = rels1.cuda()
        #drug_data.x = self.gcn_layer(drug_data)
        drug_data.x = self.mlp(drug_data.x)
        #new_feats = self.propagation_layer(drug_data)  #//mpnn new feats
        #new_feats = self.gcn_layer(drug_data)
        '''new_feats = self.Mpnn(drug_data.x,drug_data.edge_index)
        x = new_feats
        new_feats = self.Mpnn1(new_feats,drug_data.edge_index)
        new_feats = self.Mpnn2(new_feats, drug_data.edge_index)
        new_feats = self.bn(new_feats)
        new_feats = (new_feats+x).relu()
        x = new_feats
        new_feats = self.Mpnn3(new_feats, drug_data.edge_index)
        new_feats = self.Mpnn4(new_feats, drug_data.edge_index)
        new_feats = self.bn1(new_feats)
        new_feats = (new_feats+x).relu()
        drug_data.x = new_feats'''
        #drug_data.x = self.Mpnn(drug_data.x, drug_data.edge_index)
        '''drug_data.xfp = torch.reshape(drug_data.xfp, [-1, 2048])
        drug_data.xfp = drug_data.xfp.float()
        drug_data.xfp = self.mlp1(drug_data.xfp)'''
        #print(drug_data.xfp)
        drug_data.xsmpretrain = torch.reshape(drug_data.xsmpretrain,[-1,256])
        '''drug_data.xsmvec = torch.reshape(drug_data.xsmvec, [-1, 94])
        drug_data.xsmvec = drug_data.xsmvec.float()
        drug_data.xsmvec = self.mlp2(drug_data.xsmvec)'''

        #drug_data.xsmvec = drug_data.xsmvec.long()
        #drug_data.xsmvec = self.embs(drug_data.xsmvec)
        #drug_data.xsmvec = self.encoder_layer(drug_data.xsmvec)
        #drug_data.xsmvec = drug_data.xsmvec.mean(axis = 1)

        '''drug_data.x = self.Mpnn(drug_data.x,drug_data.edge_index)
        x = drug_data.x
        drug_data.x = self.Mpnn1(drug_data.x,drug_data.edge_index)
        drug_data.x = self.bn(drug_data.x)
        drug_data.x = (drug_data.x+x).relu()
        x = drug_data.x
        drug_data.x = self.Mpnn3(drug_data.x,drug_data.edge_index)
        drug_data.x = self.bn1(drug_data.x)
        drug_data.x = (drug_data.x+x).relu()'''

        drug_data.x = self.GNNML(drug_data)
        drug_data.x = global_mean_pool(drug_data.x, drug_data.batch)
        drug_data.x = self.mlp3(drug_data.x)
        drug_data.xsmpretrain = self.mlp4(drug_data.xsmpretrain)
        #drug_data.xsmvec = self.mlp4(drug_data.xsmvec)
        #多视图共享专有特征分离
        view1_specific = F.relu(self.specific1_l1(drug_data.x))
        #view1_specific = F.relu(self.specific1_l2(view1_specific))
        view1_specific = F.relu(self.specific1_l3(view1_specific))
        #print(drug_data.x.shape)
        view1_shared = self.gen1(drug_data.x)
        d1 = self.dis1(view1_shared.detach())
        tx = self.dis1(drug_data.xsmpretrain)
        '''view1_shared = F.relu(self.shared1_l1(drug_data.x))
        #view1_shared = F.relu(self.shared1_l2(view1_shared))
        view1_shared = torch.sigmoid(self.shared1_l3(view1_shared))'''

        # View2
        view2_specific = F.relu(self.specific2_l1(drug_data.xsmpretrain))
        #view2_specific = F.relu(self.specific2_l2(view2_specific))
        view2_specific = F.relu(self.specific2_l3(view2_specific))

        view2_shared = self.gen2(drug_data.xsmpretrain)
        d2 = self.dis1(view2_shared.detach())
        txsm = self.dis1(drug_data.x)
        '''view2_shared = F.relu(self.shared2_l1(drug_data.xsmvec))
        #view2_shared = F.relu(self.shared2_l2(view2_shared))
        view2_shared = torch.sigmoid(self.shared2_l3(view2_shared))'''

        view_shared = (view1_shared+view1_shared)/2
        drug_data.x = view1_specific[drug_data.batch]
        drug_data.xsmpretrain = view2_specific[drug_data.batch]
        view_shared = view_shared[drug_data.batch]
        view1_shared1 = view1_shared[drug_data.batch]
        view2_shared1 = view2_shared[drug_data.batch]


        #drug_data.x = self.GNNML2(drug_data)





        '''x = new_feats
        new_feats = self.Mpnn5(new_feats, drug_data.edge_index)
        new_feats = self.Mpnn6(new_feats, drug_data.edge_index)
        new_feats = self.bn2(new_feats)
        new_feats = (new_feats+x).relu()
        x = new_feats
        new_feats = self.Mpnn7(new_feats, drug_data.edge_index)
        new_feats = self.Mpnn8(new_feats, drug_data.edge_index)
        new_feats = self.bn3(new_feats)
        new_feats = (new_feats+x).relu()'''

        '''new_feats = self.bn2(new_feats)
        new_feats = self.Mpnn2(new_feats,drug_data.edge_index)
        x = new_feats
        new_feats = self.Mpnn3(new_feats,drug_data.edge_index)
        new_feats = self.bn1(new_feats)
        new_feats = (new_feats+x).relu()
        new_feats = self.bn3(new_feats)
        new_feats = self.Mpnn4(new_feats, drug_data.edge_index)
        x = new_feats
        new_feats = self.Mpnn5(new_feats, drug_data.edge_index)
        new_feats = self.bn4(new_feats)
        new_feats = (new_feats + x).relu()'''
        #new_feats = self.gcn_layer(drug_data)
        #print(drug_pairs.batch)
        #new_feats = self.gcn(drug_data)          #//gcn new feats
        #new_feats = self.gat(drug_data)          #//gat new feats
        #new_feats = self.GNNML1(drug_data)           #//GNNML1 new feats
        #print(new_feats.shape)
        #drug_data.x = new_feats
        #drug_data.x = self.propagation_layer(drug_data)
        '''drug_data.xchemfea = torch.reshape(drug_data.xchemfea,[-1,6])
        drug_data.x = torch.cat((drug_data.x,drug_data.xchemfea[drug_data.batch]),dim = 1)'''
        drug_data.x = torch.cat((drug_data.x,drug_data.xsmpretrain,view_shared),dim = 1)
        x_j = drug_data.x[node_j_for_pairs]
        x_i = drug_data.x[node_i_for_pairs]
        '''print(a)
        print(drug_pairs.j_indices,drug_pairs.i_indices)'''
        '''print(drug_pairs.batch)
        print(drug_pairs.batch.shape)
        print(node_j_for_pairs)
        print(node_i_for_pairs)'''
        '''print(node_j_for_pairs)
        print(node_i_for_pairs)
        print(rels)
        print(drug_pairs.edge_index)'''
        global j
        j=0
        #text_save('338 mpnn2cancha mean.csv',x_i.tolist())
        #text_save('2206 mpnn2cancha mean.csv',x_j.tolist())
        #x_i = torch.cat((x_i,x_j),0)
        a1 = global_mean_pool(x_i, drug_pairs.i_indices)
        a2 = global_mean_pool(x_j, drug_pairs.j_indices)
        #text_save('338 juhe.csv',a1.tolist())
        #text_save('2206 juhe.csv',a2.tolist())
        #scores = (a1+a2)/2
        scores = torch.cat([a1,a2],dim = 1)
        #scores = a1*a2
        torch.save(scores, 'test_save_tensor.pt')
        torch.save(rels, 'test_label.pt')
        #print(scores.shape)


        #print(scores.shape)
        #print(rels.shape)

        #scores = global_mean_pool(x_i, bat1)
        #text_save('2206&338.csv', scores.tolist())
        #print(scores.shape)

        scores = self.fclayer1(scores).relu()
        '''scores = scores.cpu().detach().numpy()
        scores = self.ts.fit_transform(scores)
        scores = torch.from_numpy(scores).cuda()'''
        #scores = self.fclayer2(scores).relu()
        scores = self.fclayer3(scores).relu()

        scores = self.fclayer(scores)
        #scores = torch.sigmoid(scores)
        pair_repr = scores.squeeze(dim=1)
        #pair_repr = scores
        #print(pair_repr)
        #print(pair_repr.shape)

        #print(scores.shape)
        #x = torch.sigmoid(self.fclayer(b))
        #print(x)
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None
        attentions = None
        x_i = x_j = None




        relschu = rels


        '''rels = self.rel_embs(rels1)
        attentions = self.co_attention_layer(x_j, x_i, drug_pairs)
        #attentions = self.co_attention_layer(x_j, x_i, drug_pairs, rels)
        pair_repr = attentions.unsqueeze(-1) * ((x_i[drug_pairs.edge_index[1]] @ self.s_pro) * (x_j[drug_pairs.edge_index[0]] @ self.s_pro))
        #print(drug_data)
        #print(drug_pairs)
        #print(attentions)
        #s s
        drug_data = new_feats = None
        node_i_for_pairs = node_j_for_pairs = None 
        attentions = None
        x_i = x_j = None
        #print(pair_repr.sum(1))
        pair_repr = scatter(pair_repr, drug_pairs.edge_index_batch, reduce='add', dim=0)
        pair_repr = pair_repr.sum(1)
        #print(pair_repr.shape)'''
        
        p_scores, n_scores = self.compute_score(pair_repr,  batch_size, rels, relschu)
        return p_scores, n_scores , view1_specific , view2_specific , view1_shared , view2_shared , d1 , d2 ,tx , txsm

    def compute_score(self, pair_repr, batch_size, rels, relschu):
        #print(len(rels))
        #rels = self.rel_proj(rels)
        #print(pair_repr.sum(-1))
        #scores = (pair_repr * rels).sum(-1)
        #scores = pair_repr.sum(-1)
        #print(scores.shape)
        '''2维'''
        '''scores=pair_repr
        p_scores=torch.empty(0,2)
        p_scores=p_scores.cuda()
        n_scores=torch.empty(0,2)
        n_scores=n_scores.cuda()
        m=0
        #print(scores.shape)
        for n in range(len(relschu)):
            if relschu[n]==1:
                p_scores=torch.cat((p_scores, scores[n:n+1]),axis=0)
            elif relschu[n]==0:
                n_scores=torch.cat((n_scores, scores[n:n+1]),axis=0)
                m=m+1
        if m==0:
            n_scores = n_scores.view(0,0)'''

        '''1维'''
        scores=pair_repr
        #print(scores)
        #print(rels)
        #print(len(pair_repr))
        #print(len(scores))
        p_scores=torch.empty(0,1)
        #print(p_scores.shape)
        p_scores=p_scores.cuda()
        n_scores=torch.empty(0)
        n_scores=n_scores.cuda()
        m=0

        for n in range(len(relschu)):
            if relschu[n]==1:
                p_scores=torch.cat((p_scores, scores[n:n+1].unsqueeze(-1)),axis=0)
            elif relschu[n]==0:
                n_scores=torch.cat((n_scores, scores[n:n+1]),axis=0)
                m=m+1
        if m==0:
            n_scores = n_scores.view(0,0,0)
        else:
            n_scores = n_scores.view(m,-1,1)
        #p_scores, n_scores = scores[:batch_size].unsqueeze(-1), scores[batch_size:].view(batch_size, -1, 1)
        #print(len(p_scores))
        #print(len(n_scores))
        #print(p_scores.shape)
        #print(n_scores.shape)
        return p_scores, n_scores

class dis1(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 因为对于D的评估应该是在【0-1】之间的数值，所以这里采用的是Sigmod激活
        )

    def forward(self , x):
        y = self.D(x)
        return y

class dis2(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 因为对于D的评估应该是在【0-1】之间的数值，所以这里采用的是Sigmod激活
        )

    def forward(self, x):
        y = self.D(x)
        return y

class gen1(nn.Module):
    def __init__(self):
        super().__init__()
        self.G1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self , x):
        y = self.G1(x)
        return y

class gen2(nn.Module):
    def __init__(self):
        super().__init__()
        self.G2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self , x):
        y = self.G2(x)
        return y


class GmpnnBlock(nn.Module):
    def __init__(self, edge_feats, n_feats, n_iter, dropout):
        super().__init__()
        self.n_feats = n_feats
        self.n_iter = n_iter
        self.dropout = dropout
        self.snd_n_feats = n_feats
        #self.snd_n_feats = n_feats * 2
        #self.snd_n_feats = 122
        #print(torch.Tensor(self.n_feats, self.n_feats))
        self.w_i = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.w_j = nn.Parameter(torch.Tensor(self.n_feats, self.n_feats))
        self.a = nn.Parameter(torch.Tensor(1, self.n_feats))
        self.bias = nn.Parameter(torch.zeros(self.n_feats))

        self.edge_emb = nn.Sequential(
            nn.Linear(edge_feats, self.n_feats)
        )

        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )

        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )

        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            CustomDropout(self.dropout),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )


        glorot(self.w_i)
        glorot(self.w_j)
        glorot(self.a)

        self.sml_mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.n_feats)
        )

        self.sml_mlp1 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_feats, self.snd_n_feats),
            nn.BatchNorm1d(self.snd_n_feats)
        )
    
    def forward(self, data):
        data.xchemfea = torch.reshape(data.xchemfea,[-1,6])
        #print(data.xchemfea[data.batch].shape)
        edge_index = data.edge_index
        edge_feats = data.edge_feats
        edge_feats = self.edge_emb(edge_feats)
        deg = degree(edge_index[1], data.x.size(0), dtype=data.x.dtype)
        #print(data.batch)
        #print(deg)
        assert len(edge_index[0]) == len(edge_feats)
        alpha_i = (data.x @ self.w_i)
        alpha_j = (data.x @ self.w_j)
        alpha = alpha_i[edge_index[1]] + alpha_j[edge_index[0]] + self.bias
        alpha = self.sml_mlp(alpha)

        assert alpha.shape == edge_feats.shape
        alpha = (alpha* edge_feats).sum(-1)
        alpha = alpha / (deg[edge_index[0]])
        edge_weights = torch.sigmoid(alpha)
        #print(edge_weights)
        assert len(edge_weights) == len(edge_index[0])
        edge_attr = data.x[edge_index[0]] * edge_weights.unsqueeze(-1)
        assert len(alpha) == len(edge_attr)
        out = edge_attr
        for _ in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + (out * edge_weights.unsqueeze(-1))
        #print(scatter(torch.tensor([[-1,0,1,2],[3,4,5,6],[7,8,9,10],[11,12,13,14]]), torch.tensor([0, 1, 2, 2]),dim = 0,reduce='mean'))
        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x1=x
        x = self.mlp(x)
        #x = torch.cat((x,data.xchemfea[data.batch]),dim = 1)
        #print(x.shape)
        '''x1=self.sml_mlp1(x1)
        y=(x+x1).relu()
        return y'''
        return x

    def mlp(self, x): 
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2

        return x

class Mpnn(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Mpnn, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 第二步：线性变换节点特征矩阵。
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, out_channels)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        return x

class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, out_channels,heads=4)
        self.conv2 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.conv3 = GCNConv(hidden_channels*4, out_channels)
        self.conv4 = GCNConv(in_channels*4,out_channels)
        self.bn = nn.BatchNorm1d(hidden_channels*4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv5 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.conv6 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.conv7 = GCNConv(hidden_channels*4, out_channels)
        self.convGCN = GCNConv(in_channels, out_channels)
        self.convGAT = GATConv(in_channels, out_channels)

    def forward(self, data):

        #x = self.conv1(data.x, data.edge_index).relu()
        x = self.convGCN(data.x, data.edge_index).relu()
        #y1=x
        #x = self.conv2(x, data.edge_index).relu()
        #x=self.bn(x)
        #y = self.conv5(y1, data.edge_index).relu()
        #x = (x+y).relu()
        #x=self.bn(x)
        #y=x

        #x = self.conv3(x, data.edge_index)

        #x = self.convGCN(data.x,data.edge_index).relu()//GCN
        #x = self.convGAT(data.x,data.edge_index).relu()

        #x=self.bn1(x)
        #y = self.conv4(y1, data.edge_index)

        #return (x+y).relu()
        '''x = self.conv1(data.x, data.edge_index).relu()
        y1=x
        x=self.conv2(x, data.edge_index).relu()
        x=self.bn(x)
        x=(y1+x).relu()
        y1 = x
        x = self.conv5(x, data.edge_index).relu()
        x=self.bn(x)
        x = (y1 + x).relu()
        y1 = x
        x = self.conv6(x, data.edge_index).relu()
        x=self.bn(x)
        x = (y1 + x).relu()
        y1 = x
        y1 = self.conv7(y1,data.edge_index)
        x = self.conv3(x, data.edge_index)
        x=self.bn1(x)
        x = (y1 + x).relu()'''
        return x


class GNNML1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GNNML1, self).__init__()

        # number of neuron
        nin1 = in_channel
        nout = out_channel
        # three part concatenate or sum?
        self.concat = False

        if self.concat:
            nin = 3 * nout
        else:
            nin = nout
        self.conv11 = SpectConv(nin1, nout, selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)
        self.conv41 = SpectConv(nin, nout, selfconn=False)

        self.fc11 = torch.nn.Linear(nin1, nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)
        self.fc41 = torch.nn.Linear(nin, nout)

        self.fc12 = torch.nn.Linear(nin1, nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)
        self.fc42 = torch.nn.Linear(nin, nout)

        self.fc13 = torch.nn.Linear(nin1, nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)
        self.fc43 = torch.nn.Linear(nin, nout)
        self.rnn_layer = nn.RNN(input_size=2048, hidden_size=128, num_layers=2)
        self.cnn = nn.Conv1d(in_channels=2048, out_channels = 128,kernel_size=1)

        '''self.fc1 = torch.nn.Linear(nin, 128)
        self.fc2 = torch.nn.Linear(32, 1)'''
        #self.fullc = torch.nn.Linear(128,2)

    def forward(self, data):
        x = data.x

        data.xchemfea = torch.reshape(data.xchemfea,[-1,6])
        edge_index = data.edge_index
        edge_attr = data.edge_feats
        #edge_attr = torch.ones(edge_index.shape[1], 1).to('cuda')

        if self.concat:
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index, edge_attr)),
                           F.relu(self.fc12(x) * self.fc13(x))], 1)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index, edge_attr)),
                           F.relu(self.fc22(x) * self.fc23(x))], 1)
            #x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index, edge_attr)),
                           #F.relu(self.fc32(x) * self.fc33(x))], 1)
            #x = torch.cat([F.relu(self.fc41(x)), F.relu(self.conv41(x, edge_index, edge_attr)),
                           #F.relu(self.fc42(x) * self.fc43(x))], 1)
        else:

            x = F.relu(self.fc11(x) + self.conv11(x, edge_index, edge_attr) + self.fc12(x) * self.fc13(x))
            x = F.relu(self.fc21(x) + self.conv21(x, edge_index, edge_attr) + self.fc22(x) * self.fc23(x))
            #x = F.relu(self.fc31(x) + self.conv31(x, edge_index, edge_attr) + self.fc32(x) * self.fc33(x))
            #x = F.relu(self.fc41(x) + self.conv41(x, edge_index, edge_attr) + self.fc42(x) * self.fc43(x))
        #print(data.batch)
        '''b = global_add_pool(x, data.batch)
        print(b.shape)
        print(x.shape)
        
        #print(data.batch.shape)
        #print(x.shape)
        b = F.relu(self.fullc(b))'''
        #print(data.batch.shape)
        #print(data.batch)
        #x = F.relu(self.fc1(x))
        #print(x)
        #x = torch.cat((x,data.xchemfea[data.batch]),dim = 1)
        return x
        #return self.fc2(x)


class GNNML1PRO(nn.Module):
    def __init__(self):
        super(GNNML1PRO, self).__init__()

        S = 1
        nout1 = 64
        nout2 = 64
        nout3 = 16
        nin = nout1 + nout2 + nout3

        self.bn1 = torch.nn.BatchNorm1d(nin)
        self.bn2 = torch.nn.BatchNorm1d(nin)

        self.conv11 = SpectConv(64, nout2, S, selfconn=False)
        self.conv21 = SpectConv(nin, nout2, S, selfconn=False)

        self.fc11 = torch.nn.Linear(64, nout1)
        self.fc21 = torch.nn.Linear(nin, nout1)

        self.fc12 = torch.nn.Linear(64, nout3)
        self.fc22 = torch.nn.Linear(nin, nout3)

        self.fc13 = torch.nn.Linear(64, nout3)
        self.fc23 = torch.nn.Linear(nin, nout3)

        self.fc2 = torch.nn.Linear(nin, 128)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_feats
        #edge_attr = torch.ones(edge_index.shape[1], 1).to('cuda')

        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index, edge_attr)),
                       F.relu(self.fc12(x)) * F.relu(self.fc13(x))], 1)
        # x=self.bn1(x)

        x = F.dropout(x, p=0.1, training=self.training)
        x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index, edge_attr)),
                       F.relu(self.fc22(x)) * F.relu(self.fc23(x))], 1)
        # x=self.bn2(x)
        #x = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], 1)
        # x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class GNNML3(nn.Module):
    def __init__(self):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1 = 64
        nout2 = 64

        nin = nout1 + nout2
        ne = 6
        ninp = 64

        self.conv1 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=ninp, nout1=nout1, nout2=nout2)
        self.conv2 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=nin, nout1=nout1, nout2=nout2)
        self.conv3 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=nin, nout1=nout1, nout2=nout2)
        self.conv4 = ML3Layer(learnedge=True, nedgeinput=ne, nedgeoutput=ne, ninp=nin, nout1=nout1, nout2=nout2)

        self.fc1 = torch.nn.Linear(nin, 128)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_feats

        x = (self.conv1(x, edge_index, edge_attr))
        #x = (self.conv2(x, edge_index, edge_attr))
        #x = (self.conv3(x, edge_index, edge_attr))
        #x = (self.conv4(x, edge_index, edge_attr))

        #x = global_add_pool(x, data.batch)
        #x = F.relu(self.fc1(x))
        return x
        #return self.fc2(x)


class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = (lambda x: x ) if p == 0 else nn.Dropout(p)
    
    def forward(self, input):
        return self.dropout(input)
