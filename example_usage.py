import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ranking_mse_loss import (
    CombinedMSERankingLoss, 
    RankingLoss, 
    AdaptiveRankingWeight,
    ListwiseRankingLoss,
    PairwiseRankingLoss
)


# Simple neural network for demonstration
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super(SimpleRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()


def generate_sample_data(n_samples: int = 1000, input_dim: int = 10):
    """Generate sample regression data for demonstration."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    # Create target with some structure that benefits from ranking
    y = torch.sum(X[:, :3] ** 2, dim=1) + 0.1 * torch.randn(n_samples)
    return X, y


def train_with_combined_loss():
    """Example: Training with combined MSE + Ranking loss."""
    print("=== Training with Combined MSE + Ranking Loss ===")
    
    # Generate data
    X_train, y_train = generate_sample_data(800, 10)
    X_val, y_val = generate_sample_data(200, 10)
    
    # Create model and loss function
    model = SimpleRegressor(input_dim=10)
    combined_loss = CombinedMSERankingLoss(ranking_weight=0.1, ranking_margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train)
        total_loss, mse_loss, ranking_loss = combined_loss(predictions, y_train)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, "
                  f"MSE = {mse_loss:.4f}, Ranking = {ranking_loss:.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_total_loss, val_mse_loss, val_ranking_loss = combined_loss(val_predictions, y_val)
        print(f"Validation: Total Loss = {val_total_loss:.4f}, "
              f"MSE = {val_mse_loss:.4f}, Ranking = {val_ranking_loss:.4f}")
    
    return model, val_mse_loss.item()


def train_with_adaptive_ranking():
    """Example: Training with adaptive ranking weight."""
    print("\n=== Training with Adaptive Ranking Weight ===")
    
    # Generate data
    X_train, y_train = generate_sample_data(800, 10)
    X_val, y_val = generate_sample_data(200, 10)
    
    # Create model and components
    model = SimpleRegressor(input_dim=10)
    mse_loss = nn.MSELoss()
    ranking_loss = RankingLoss(margin=1.0)
    adaptive_weight = AdaptiveRankingWeight(initial_weight=0.5, decay_factor=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train)
        mse = mse_loss(predictions, y_train)
        ranking = ranking_loss(predictions, y_train)
        
        # Combined loss with adaptive weight
        weight = adaptive_weight.get_weight()
        total_loss = mse + weight * ranking
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update adaptive weight
        adaptive_weight.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}, "
                  f"MSE = {mse:.4f}, Ranking = {ranking:.4f}, Weight = {weight:.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_mse = mse_loss(val_predictions, y_val)
        val_ranking = ranking_loss(val_predictions, y_val)
        print(f"Validation: MSE = {val_mse:.4f}, Ranking = {val_ranking:.4f}")
    
    return model, val_mse.item()


def compare_ranking_methods():
    """Compare different ranking loss implementations."""
    print("\n=== Comparing Different Ranking Methods ===")
    
    # Generate data
    X_train, y_train = generate_sample_data(800, 10)
    X_val, y_val = generate_sample_data(200, 10)
    
    ranking_methods = {
        'Standard': RankingLoss(margin=1.0),
        'Pairwise': PairwiseRankingLoss(margin=1.0),
        'Listwise': ListwiseRankingLoss(temperature=1.0)
    }
    
    results = {}
    
    for method_name, ranking_loss_fn in ranking_methods.items():
        print(f"\nTraining with {method_name} Ranking Loss:")
        
        # Create fresh model
        model = SimpleRegressor(input_dim=10)
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):  # Shorter training for comparison
            optimizer.zero_grad()
            
            predictions = model(X_train)
            mse = mse_loss(predictions, y_train)
            ranking = ranking_loss_fn(predictions, y_train)
            total_loss = mse + 0.1 * ranking
            
            total_loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_mse = mse_loss(val_predictions, y_val)
            val_ranking = ranking_loss_fn(val_predictions, y_val)
            
        results[method_name] = val_mse.item()
        print(f"  Validation MSE: {val_mse:.4f}, Ranking: {val_ranking:.4f}")
    
    print(f"\nResults Summary:")
    for method, mse in results.items():
        print(f"  {method}: {mse:.4f}")
    
    return results


def baseline_mse_only():
    """Baseline training with MSE loss only."""
    print("\n=== Baseline: MSE Loss Only ===")
    
    # Generate data
    X_train, y_train = generate_sample_data(800, 10)
    X_val, y_val = generate_sample_data(200, 10)
    
    # Create model
    model = SimpleRegressor(input_dim=10)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        predictions = model(X_train)
        loss = mse_loss(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: MSE Loss = {loss:.4f}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_mse = mse_loss(val_predictions, y_val)
        print(f"Validation MSE: {val_mse:.4f}")
    
    return model, val_mse.item()


if __name__ == "__main__":
    # Run all examples
    print("Demonstrating Ranking Loss for MSE Regularization\n")
    
    # Baseline
    baseline_model, baseline_mse = baseline_mse_only()
    
    # Combined loss
    combined_model, combined_mse = train_with_combined_loss()
    
    # Adaptive ranking
    adaptive_model, adaptive_mse = train_with_adaptive_ranking()
    
    # Compare methods
    comparison_results = compare_ranking_methods()
    
    print(f"\n=== Final Comparison ===")
    print(f"Baseline (MSE only): {baseline_mse:.4f}")
    print(f"Combined MSE+Ranking: {combined_mse:.4f}")
    print(f"Adaptive Ranking: {adaptive_mse:.4f}")
    
    improvement_combined = ((baseline_mse - combined_mse) / baseline_mse) * 100
    improvement_adaptive = ((baseline_mse - adaptive_mse) / baseline_mse) * 100
    
    print(f"\nImprovement over baseline:")
    print(f"Combined: {improvement_combined:.2f}%")
    print(f"Adaptive: {improvement_adaptive:.2f}%")