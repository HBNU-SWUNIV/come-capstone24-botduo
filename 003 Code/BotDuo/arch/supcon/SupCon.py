import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
 
    def forward(self, features, labels):
        cover_features = features[labels == 0]
        cover_features = cover_features.mean(axis=1)

        cover_features = F.normalize(cover_features, p=2, dim=0)
        cover_features = cover_features.unsqueeze(0)
 
        # spoof feature (aggregation)
        stego_features = features[labels == 1]
        stego_features, _ = stego_features.max(axis=1)    
        stego_features = F.normalize(stego_features, p=2, dim=0)    
        stego_features = stego_features.unsqueeze(0)
 
        # calculate inner products
        cover_similarity_matrix = torch.matmul(cover_features, cover_features.T) / (self.temperature + 1e-8)
        cover_stego_similarity_matrix = torch.matmul(cover_features.T, stego_features) / (self.temperature + 1e-8)
        stego_similarity_matrix = torch.matmul(stego_features, stego_features.T)  / (self.temperature + 1e-8)
 
        # for numerical stability
        sim_max_live, _ = torch.max(cover_similarity_matrix, dim=1, keepdim=True)
        sim_max_spoof, _ = torch.max(stego_similarity_matrix, dim=1, keepdim=True)
 
        sim_max = torch.max(sim_max_live, sim_max_spoof).detach()
 
        cover_similarity_matrix = cover_similarity_matrix - sim_max
        stego_similarity_matrix = stego_similarity_matrix - sim_max
        cover_stego_similarity_matrix = cover_stego_similarity_matrix - sim_max
 
        # calc log prob
        exp_cover_similarity = torch.exp(cover_similarity_matrix)
        exp_stego_similarity = torch.exp(stego_similarity_matrix)
        exp_cover_stego_similarity = torch.exp(cover_stego_similarity_matrix)
 
        denom = exp_cover_similarity.sum(1, keepdim=True) + exp_stego_similarity.sum(1, keepdim=True) + exp_cover_stego_similarity.sum(1, keepdim=True)
 
        log_prob_cover = cover_similarity_matrix - torch.log(denom + 1e-8)
        log_prob_stego = stego_similarity_matrix - torch.log(denom + 1e-8)
 
        # Normalize loss
        loss = -(log_prob_cover.mean() + log_prob_stego.mean()) / 2
 
        return loss