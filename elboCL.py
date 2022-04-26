import torch
from torch import nn
import torch.nn.functional as F


class ELBO(nn.Module):
    def __init__(self, num_class, num_cluster, feat_dim, tau, kappa, eta, device):
        super(ELBO, self).__init__()

        print("--------------------------Initializing -----------------------------------")
        self.num_class = num_class
        self.num_cluster = num_cluster
        self.feat_dim = feat_dim
        self.tau = tau
        self.kappa = kappa
        self.eta = eta
        self.device = device

        self.prototype = nn.Parameter(torch.nn.init.uniform_(torch.Tensor(self.feat_dim, self.num_cluster), a = 0, b = 1))
        self.logSoftmax = torch.nn.LogSoftmax(dim=1)
        print("----------------------Initialization Ends-----------------------------------")


    def update_cluster(self, new_center):
        with torch.no_grad():
            out_ids = torch.arange(self.num_cluster).to(self.device)
            out_ids = out_ids.long()  # BS x 1
            self.prototype.index_copy_(1, out_ids, new_center.T)


    def forward(self, emb, emb2, y):
        features = torch.cat((emb, emb2), dim=0)
        batchSize = features.shape[0]
        y = y.contiguous().view(-1, 1)
        mask = torch.eq(y, y.T).float().to(self.device)
        mask = mask.repeat(2, 2)

        anchor_dot_cluster = torch.matmul(features, self.prototype) # BS x M
        anchor_dot_contrast = torch.matmul(features, features.T) # BS x BS

        # clusterscl loss
        pi_logit = torch.div(anchor_dot_cluster, self.kappa)
        log_pi = self.logSoftmax(pi_logit + 1e-18) # BS x M
        pi = torch.exp(log_pi)  # cluster distribution p(c | v), BS x M

        loss_0 = torch.mean(torch.sum(pi * log_pi, dim=1))

        # compute the alignment with the augmented positives and negatives
        align_cluster = anchor_dot_cluster.T.view(self.num_cluster, batchSize, 1).repeat(1, 1, batchSize) 
        align_contrast = anchor_dot_contrast.repeat(self.num_cluster, 1).view(self.num_cluster, batchSize, batchSize)
        weight1 = torch.div(torch.exp(align_cluster), (torch.exp(align_cluster) + torch.exp(align_contrast)))
        weight2 = torch.div(torch.exp(align_contrast), (torch.exp(align_cluster) + torch.exp(align_contrast)))
        
        anchor_dot_augmentation = (weight1 * align_cluster + weight2 * align_contrast) / self.tau + 1e-18 # M x BS x BS

        logits_max, _ = torch.max(anchor_dot_augmentation, dim=2, keepdim=True)
        logits = anchor_dot_augmentation - logits_max.detach()
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batchSize).view(-1, 1).to(self.device),
            0
        )
        # set the diagonal elements to be 0
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_logits = logits - torch.log(exp_logits.sum(2, keepdim=True))
        normalized_logits = torch.exp(log_logits) # p(s | v, c)
        # remain the prob of each anchor's positives
        log_logits_pos = torch.mul(log_logits, mask)
        normalized_logits_pos = torch.mul(normalized_logits, mask)
        del log_logits
        del normalized_logits
        
        pi_normalized_logits_pos = pi.T.view(self.num_cluster, batchSize, 1) * normalized_logits_pos
        posterior = torch.div(pi_normalized_logits_pos, torch.add(torch.sum(pi_normalized_logits_pos, 0), 1 - mask))
        posterior = torch.mul(posterior, mask) # q(c | v, s)

        loss = -torch.mean(torch.div(torch.sum(torch.sum(posterior * (log_pi.T.view(self.num_cluster, batchSize, 1) + log_logits_pos - torch.log(posterior + 1e-18)), 0), 1), torch.sum(mask, 1)))

        loss_final = loss + self.eta * loss_0

        return loss_final