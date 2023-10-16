import torch
from torch import nn
import torch.nn.functional as F



class SigmoidLoss(nn.Module):

    '''def forward(self,npbatch_probas_pred,npbatch_ground_truth):
        npbatch_ground_truth = npbatch_ground_truth.long()
        loss = F.cross_entropy(npbatch_probas_pred,npbatch_ground_truth)
        loss2 = torch.mean(torch.nn.CrossEntropyLoss(reduce=None)(npbatch_probas_pred, npbatch_ground_truth.long()))
        loss3 = torch.mean(torch.nn.NLLLoss(reduce=False)(torch.nn.LogSoftmax(dim=-1)(npbatch_probas_pred), npbatch_ground_truth.long()))
        loss.requires_grad_(True)
        return loss'''
    def orthogonal_loss(self,shared, specific):
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.t().matmul(specific)
        cost = correlation_matrix.matmul(correlation_matrix).mean()
        cost = F.relu(cost)
        #print(cost)
        return cost
    def dot_product_normalize(self,shared_1, shared_2):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        num_of_samples = shared_1.size(0)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        # Dot product
        match_map = torch.bmm(shared_1.view(num_of_samples, 1, -1), shared_2.view(num_of_samples, -1, 1))
        mean = match_map.mean()
        return mean

    def forward(self, p_scores, n_scores,view1_specific , view2_specific , view1_shared , view2_shared):
        # Similarity Loss
        #similarity_loss = - self.dot_product_normalize(view1_shared, view2_shared)
        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(view1_shared, view1_specific)
        orthogonal_loss2 = self.orthogonal_loss(view2_shared, view2_specific)

        orthogonal_loss3 = self.orthogonal_loss(view1_specific, view2_specific)

        # Classification Loss
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        a=p_loss*len(n_scores)/(len(p_scores)+len(n_scores))
        b=n_loss*len(p_scores)/(len(p_scores)+len(n_scores))
        '''if(len(p_scores)==0):
            c=b
        elif(len(n_scores)==0):
            c=a
        else:c=a+b
        return c, p_loss, n_loss'''
        if(len(p_scores)==0):
            c=n_loss
        elif(len(n_scores)==0):
            c=p_loss
        else:c=(p_loss + n_loss) / 2
        c = c + orthogonal_loss1 * 0.2 + orthogonal_loss2 * 0.2 + orthogonal_loss3 * 0.2
        #c = c + orthogonal_loss1 * 0.2 + orthogonal_loss2 * 0.2 + similarity_loss * 0.2

        #c = c + abs(orthogonal_loss1*0.2) + abs(orthogonal_loss2*0.2)  +abs(similarity_loss*0.2)
        return c, p_loss, n_loss
        #return (p_loss + n_loss) / 2, p_loss, n_loss

    '''def forward(self,score,target):
        score=torch.from_numpy(score)
        score = torch.unsqueeze(score,dim=-1)
        #print(score.dtype)
        target=torch.from_numpy(target)
        #target = torch.unsqueeze(target,dim=-1)
        target=target.long()
        #print(target.dtype)
        loss = F.cross_entropy(score,target)
        return loss'''

