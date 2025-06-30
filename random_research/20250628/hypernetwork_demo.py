import torch
import torch.nn as nn
import torch.optim as optim


# === Hypernetwork ===
class HyperNetwork(nn.Module):
    def __init__(self, hyper_input_dim, target_in_dim, target_out_dim):
        super().__init__()
        # This network will generate weights & bias for the target linear layer
        hidden_dim = 64
        self.generator = nn.Sequential(
            nn.Linear(hyper_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_in_dim * target_out_dim + target_out_dim)
        )
        self.target_in_dim = target_in_dim
        self.target_out_dim = target_out_dim

    def forward(self, z):
        # Generate a flat vector containing weights + biases
        generated = self.generator(z)
        W_flat = generated[:, :self.target_in_dim * self.target_out_dim]
        b_flat = generated[:, self.target_in_dim * self.target_out_dim:]
        # Reshape weight to matrix
        W = W_flat.view(-1, self.target_out_dim, self.target_in_dim)
        b = b_flat.view(-1, self.target_out_dim)
        return W, b


# === Dummy dataset ===
torch.manual_seed(0)
X = torch.randn(100, 3)
true_W = torch.tensor([[2.0, -1.0, 0.5]])
true_b = torch.tensor([0.3])
y = X @ true_W.T + true_b + 0.1 * torch.randn(100, 1)

# === Initialize hypernetwork ===
hypernet = HyperNetwork(hyper_input_dim=4, target_in_dim=3, target_out_dim=1)
optimizer = optim.Adam(hypernet.parameters(), lr=1e-3)

# Dummy hyper-input (e.g., could represent task embedding or context)
z = torch.randn(1, 4)

# === Training loop ===
for step in range(1000):
    optimizer.zero_grad()

    # Generate weights for target network from hypernetwork
    W, b = hypernet(z)  # W: [1, 1, 3], b: [1, 1]

    # Apply target network manually
    y_pred = X @ W[0].T + b[0]  # shape [100, 1]

    loss = nn.MSELoss()(y_pred, y)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss {loss.item():.6f}")

# Final learned weights
print("\nGenerated weights after training:")
print("W:", W[0].detach().numpy())
print("b:", b[0].detach().numpy())
