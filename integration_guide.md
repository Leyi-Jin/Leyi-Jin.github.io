# Ranking Loss Integration Guide

This guide shows you how to integrate ranking loss as regularization for your existing MSE loss function.

## Quick Start

### Option 1: Use the Combined Loss Function (Easiest)

Replace your existing MSE loss with the combined loss:

```python
from ranking_mse_loss import CombinedMSERankingLoss

# Instead of:
# criterion = nn.MSELoss()

# Use:
criterion = CombinedMSERankingLoss(
    ranking_weight=0.1,    # λ parameter (adjust as needed)
    ranking_margin=1.0,    # margin for ranking loss
    reduction='mean'
)

# In your training loop:
total_loss, mse_loss, ranking_loss = criterion(predictions, targets)
total_loss.backward()

# Optional: Log individual components
print(f"MSE: {mse_loss:.4f}, Ranking: {ranking_loss:.4f}")
```

### Option 2: Manual Combination (More Control)

Keep your existing MSE loss and add ranking loss manually:

```python
from ranking_mse_loss import RankingLoss

# Keep your existing MSE loss
mse_criterion = nn.MSELoss()

# Add ranking loss
ranking_criterion = RankingLoss(margin=1.0)
ranking_weight = 0.1  # λ parameter

# In your training loop:
mse_loss = mse_criterion(predictions, targets)
ranking_loss = ranking_criterion(predictions, targets)
total_loss = mse_loss + ranking_weight * ranking_loss

total_loss.backward()
```

### Option 3: Adaptive Weighting

Use adaptive weighting that decreases the ranking loss influence over time:

```python
from ranking_mse_loss import RankingLoss, AdaptiveRankingWeight

mse_criterion = nn.MSELoss()
ranking_criterion = RankingLoss(margin=1.0)
adaptive_weight = AdaptiveRankingWeight(
    initial_weight=0.5,    # Start with higher ranking influence
    decay_factor=0.95,     # Decay factor per epoch
    min_weight=0.01        # Minimum weight to maintain
)

# In your training loop (for each epoch):
for epoch in range(num_epochs):
    # ... training code ...
    
    mse_loss = mse_criterion(predictions, targets)
    ranking_loss = ranking_criterion(predictions, targets)
    weight = adaptive_weight.get_weight()
    total_loss = mse_loss + weight * ranking_loss
    
    total_loss.backward()
    
    # Update weight at end of epoch
    adaptive_weight.step()
```

## Key Parameters to Tune

### 1. Ranking Weight (λ)
- **Default**: 0.1
- **Range**: 0.01 - 1.0
- **Effect**: Controls the influence of ranking loss vs MSE loss
- **Tuning**: Start with 0.1, increase if you want stronger ranking constraints

### 2. Ranking Margin
- **Default**: 1.0
- **Range**: 0.5 - 2.0
- **Effect**: Controls how strict the ranking constraints are
- **Tuning**: Larger margins = stricter ranking requirements

### 3. Reduction Method
- **Options**: 'mean', 'sum', 'none'
- **Default**: 'mean'
- **Effect**: How to aggregate the loss across batch

## Different Ranking Loss Types

### Standard Ranking Loss (Recommended)
```python
ranking_loss = RankingLoss(margin=1.0)
```
- Best for most use cases
- Efficient computation
- Good balance of ranking preservation

### Pairwise Ranking Loss
```python
from ranking_mse_loss import PairwiseRankingLoss
ranking_loss = PairwiseRankingLoss(margin=1.0)
```
- More explicit pairwise comparisons
- Slower for large batches
- Use when ranking relationships are critical

### Listwise Ranking Loss
```python
from ranking_mse_loss import ListwiseRankingLoss
ranking_loss = ListwiseRankingLoss(temperature=1.0)
```
- Based on probability distributions
- Good for ranking entire lists
- Use when global ranking order matters

## Integration Examples

### For PyTorch Lightning

```python
import pytorch_lightning as pl
from ranking_mse_loss import CombinedMSERankingLoss

class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = YourModel()
        self.criterion = CombinedMSERankingLoss(ranking_weight=0.1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x)
        total_loss, mse_loss, ranking_loss = self.criterion(predictions, y)
        
        # Log components
        self.log('train_total_loss', total_loss)
        self.log('train_mse_loss', mse_loss)
        self.log('train_ranking_loss', ranking_loss)
        
        return total_loss
```

### For Hugging Face Transformers

```python
from transformers import Trainer
from ranking_mse_loss import CombinedMSERankingLoss

class RankingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ranking_criterion = CombinedMSERankingLoss(ranking_weight=0.1)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze()
        
        total_loss, mse_loss, ranking_loss = self.ranking_criterion(predictions, labels)
        
        return (total_loss, outputs) if return_outputs else total_loss
```

### For Scikit-learn Style API

```python
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from ranking_mse_loss import CombinedMSERankingLoss

class RankingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ranking_weight=0.1, epochs=100, lr=0.001):
        self.ranking_weight = ranking_weight
        self.epochs = epochs
        self.lr = lr
    
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        criterion = CombinedMSERankingLoss(ranking_weight=self.ranking_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor).squeeze()
            total_loss, _, _ = criterion(predictions, y_tensor)
            total_loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            return self.model(X_tensor).squeeze().numpy()
```

## Benefits of Ranking Loss

1. **Better Generalization**: Encourages the model to preserve relative ordering
2. **Robustness**: Less sensitive to outliers compared to pure MSE
3. **Ranking Preservation**: Maintains important ordering relationships in data
4. **Regularization**: Acts as additional regularization to prevent overfitting

## When to Use Ranking Loss

- **Recommendation Systems**: When relative preferences matter
- **Time Series**: When trend preservation is important
- **Financial Modeling**: When ranking of stocks/assets matters
- **Quality Assessment**: When relative quality ordering is crucial
- **Any Regression**: Where relative ordering provides additional signal

## Hyperparameter Tuning Tips

1. **Start Small**: Begin with ranking_weight=0.1
2. **Monitor Both Losses**: Track MSE and ranking loss separately
3. **Validation-Based**: Use validation performance to tune parameters
4. **Gradual Increase**: Increase ranking weight if MSE doesn't degrade
5. **Dataset Dependent**: Optimal parameters vary by dataset characteristics

## Troubleshooting

### High Ranking Loss
- Reduce ranking margin
- Decrease ranking weight
- Check data distribution

### No Improvement
- Increase ranking weight
- Verify data has meaningful ordering
- Try different ranking loss types

### Training Instability
- Reduce learning rate
- Use gradient clipping
- Decrease ranking weight