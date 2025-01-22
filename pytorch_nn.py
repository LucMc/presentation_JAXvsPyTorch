import torch
import torch.nn as nn
import numpy as np
import time

# In PyTorch, we create classes that inherit from nn.Module
# Unlike JAX, which uses functional programming and pure functions
class SineNet(nn.Module):
    def __init__(self, hidden_size=32):
        # PyTorch requires calling super().__init__() in the constructor
        # JAX doesn't have this requirement as it doesn't use class-based models
        super().__init__()
        
        # PyTorch uses nn.Sequential for layer composition
        # Unlike JAX, which typically uses function composition with stax or explicit layers
        self.network = nn.Sequential(
           nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    # PyTorch models define a forward method
    # JAX uses pure functions that take parameters as arguments
    def forward(self, x):
        return self.network(x)

# Generate training data
x = torch.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
y = torch.sin(x)

# Create model instance
# In PyTorch, model instantiation creates and manages parameters
# In JAX, we'd explicitly initialize parameters using a PRNG key
model = SineNet()

# PyTorch optimizers wrap the model's parameters
# Unlike JAX optimizers which are typically functional and return updated parameter values
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
# PyTorch handles gradient computation and updates in-place
# JAX uses pure functions and explicit gradient computation
start = time.time()
for epoch in range(4000):
    # PyTorch requires zeroing gradients each iteration
    # JAX doesn't need this as it uses pure functions
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    # PyTorch accumulates gradients in the computational graph
    # JAX would use jax.value_and_grad to get both loss and gradients
    loss = nn.MSELoss()(y_pred, y)
    
    # Backward pass and optimization
    # PyTorch updates parameters in-place
    # JAX would return new parameter values
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
print(f"Finished training in {time.time()-start} seconds")
# Evaluation
model.eval()  # Switch to evaluation mode (affects dropout, batch norm)
# JAX doesn't have separate train/eval modes - you'd pass a is_training flag

x_test = torch.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_test = torch.sin(x_test)

with torch.no_grad():  # Disable gradient tracking for inference
    # JAX doesn't need this as it naturally doesn't track gradients
    y_pred = model(x_test)

# Plot results
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') # Because I'm running from terminal, change to whatever backend you want 


plt.plot(x_test.numpy(), y_test.numpy(), label='True')
plt.plot(x_test.numpy(), y_pred.numpy(), label='Predicted')
plt.legend()
plt.show()
