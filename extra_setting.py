import torch
import torch.nn as nn
import torch.nn.functional as F

def opposite_loss(predicted_labels, predicted_hardness_scores, target, criterion):

    cross_entropy_loss = criterion(predicted_labels, target).squeeze()
    cross_entropy_loss = (-1) * cross_entropy_loss
    p_i_c = torch.exp(cross_entropy_loss)

    term1 = p_i_c * torch.log(1 - predicted_hardness_scores)
    term2 = (1 - p_i_c) * torch.log(predicted_hardness_scores)
    final_loss = - (term1 + term2)
    return torch.mean(final_loss)
