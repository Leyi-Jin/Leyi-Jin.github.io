import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RankingLoss(nn.Module):
    """
    Ranking loss implementation for regularizing MSE loss.
    
    This loss encourages the model to maintain relative ordering between samples
    based on their target values, which can improve generalization.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            margin: Margin for the ranking loss (default: 1.0)
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum' (default: 'mean')
        """
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss.
        
        Args:
            predictions: Model predictions of shape (batch_size,)
            targets: Ground truth targets of shape (batch_size,)
            
        Returns:
            Ranking loss value
        """
        batch_size = predictions.size(0)
        
        if batch_size < 2:
            # Need at least 2 samples for ranking
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Create pairwise differences
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)  # (batch_size, batch_size)
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # (batch_size, batch_size)
        
        # Only consider pairs where targets have different values
        valid_pairs = (target_diff != 0).float()
        
        # Sign of target difference indicates desired ranking order
        target_sign = torch.sign(target_diff)
        
        # Ranking loss: max(0, margin - target_sign * pred_diff)
        loss_matrix = F.relu(self.margin - target_sign * pred_diff) * valid_pairs
        
        # Remove diagonal elements (self-comparisons)
        mask = torch.eye(batch_size, device=predictions.device)
        loss_matrix = loss_matrix * (1 - mask)
        
        if self.reduction == 'mean':
            # Average over all valid pairs
            num_valid_pairs = valid_pairs.sum() - batch_size  # Exclude diagonal
            if num_valid_pairs > 0:
                return loss_matrix.sum() / num_valid_pairs
            else:
                return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        elif self.reduction == 'sum':
            return loss_matrix.sum()
        else:  # 'none'
            return loss_matrix


class CombinedMSERankingLoss(nn.Module):
    """
    Combined loss function that adds ranking loss as regularization to MSE loss.
    
    Total Loss = MSE Loss + λ * Ranking Loss
    """
    
    def __init__(self, 
                 ranking_weight: float = 0.1,
                 ranking_margin: float = 1.0,
                 reduction: str = 'mean'):
        """
        Args:
            ranking_weight: Weight for the ranking loss term (λ)
            ranking_margin: Margin for the ranking loss
            reduction: Reduction method for both losses
        """
        super(CombinedMSERankingLoss, self).__init__()
        self.ranking_weight = ranking_weight
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.ranking_loss = RankingLoss(margin=ranking_margin, reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined MSE + Ranking loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Tuple of (total_loss, mse_loss, ranking_loss)
        """
        mse_loss = self.mse_loss(predictions, targets)
        ranking_loss = self.ranking_loss(predictions, targets)
        total_loss = mse_loss + self.ranking_weight * ranking_loss
        
        return total_loss, mse_loss, ranking_loss


class AdaptiveRankingWeight:
    """
    Adaptive weighting scheme for ranking loss that adjusts during training.
    """
    
    def __init__(self, 
                 initial_weight: float = 0.1,
                 decay_factor: float = 0.95,
                 min_weight: float = 0.01):
        """
        Args:
            initial_weight: Starting weight for ranking loss
            decay_factor: Factor to decay weight each epoch
            min_weight: Minimum weight to maintain
        """
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.epoch = 0
    
    def get_weight(self) -> float:
        """Get current ranking weight."""
        return self.current_weight
    
    def step(self):
        """Update weight for next epoch."""
        self.epoch += 1
        self.current_weight = max(
            self.min_weight,
            self.initial_weight * (self.decay_factor ** self.epoch)
        )
    
    def reset(self):
        """Reset to initial weight."""
        self.current_weight = self.initial_weight
        self.epoch = 0


# Alternative ranking loss implementations

class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss using ListNet approach.
    """
    
    def __init__(self, temperature: float = 1.0):
        super(ListwiseRankingLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute listwise ranking loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Listwise ranking loss
        """
        # Convert to probabilities using softmax
        pred_probs = F.softmax(predictions / self.temperature, dim=0)
        target_probs = F.softmax(targets / self.temperature, dim=0)
        
        # KL divergence between target and predicted distributions
        loss = F.kl_div(pred_probs.log(), target_probs, reduction='batchmean')
        
        return loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss using margin-based approach.
    """
    
    def __init__(self, margin: float = 1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise ranking loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Pairwise ranking loss
        """
        batch_size = predictions.size(0)
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if targets[i] != targets[j]:
                    # Determine which should be ranked higher
                    if targets[i] > targets[j]:
                        # predictions[i] should be > predictions[j]
                        loss += F.relu(self.margin - (predictions[i] - predictions[j]))
                    else:
                        # predictions[j] should be > predictions[i]
                        loss += F.relu(self.margin - (predictions[j] - predictions[i]))
                    count += 1
        
        return loss / max(count, 1)