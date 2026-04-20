import torch                          # Core PyTorch tensor library
import torch.nn as nn                 # Neural network components (layers, activations)
import torch.optim as optim           # Optimizers like Adam

# -----------------------------
# 1. Define the model
# -----------------------------
class SimpleNN(nn.Module):            # All models subclass nn.Module
    def __init__(self):
        super().__init__()            # Initialize base class

        # Define layers (these have learnable parameters)
        self.fc1 = nn.Linear(10, 32)  # Dense layer: 10 → 32
        self.relu = nn.ReLU()         # Activation function
        self.fc2 = nn.Linear(32, 1)   # Dense layer: 32 → 1

    def forward(self, x):
        # Define forward pass (data flow)
        x = self.fc1(x)               # Linear transformation
        x = self.relu(x)              # Non-linearity
        x = self.fc2(x)               # Final output layer
        return x                      # Return prediction


# -----------------------------
# 2. Create model, loss, optimizer
# -----------------------------
model = SimpleNN()                   # Instantiate model (randomly initialized weights)

criterion = nn.MSELoss()             # Loss function (mean squared error)

optimizer = optim.Adam(              # Adam optimizer
    model.parameters(),              # Pass all model parameters
    lr=0.001                         # Learning rate
)


# -----------------------------
# 3. Generate dummy data
# -----------------------------
# We'll create synthetic data where the target is a simple function of inputs
# This lets you actually see learning happen

torch.manual_seed(0)                 # For reproducibility

x = torch.randn(256, 10)             # 256 samples, each with 10 features
true_weights = torch.randn(10, 1)    # "Ground truth" linear relationship
y = x @ true_weights + 0.1 * torch.randn(256, 1)  # Add small noise


# -----------------------------
# 4. Training loop
# -----------------------------
epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()            # Clear old gradients

    outputs = model(x)               # Forward pass: predictions

    loss = criterion(outputs, y)     # Compute loss

    loss.backward()                 # Backprop: compute gradients

    optimizer.step()                # Update weights using Adam

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# -----------------------------
# 5. Test the model
# -----------------------------
with torch.no_grad():                # Disable gradient tracking (faster, no memory overhead)
    test_input = torch.randn(1, 10)  # Single test sample
    prediction = model(test_input)   # Model prediction

print("\nTest input:", test_input)
print("Model prediction:", prediction)