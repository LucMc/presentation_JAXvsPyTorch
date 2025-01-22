import torch
import torch.nn as nn
import numpy as np
import time

class SineNet(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
           nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Generate training data
x = torch.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
y = torch.sin(x)

# Create model instance
model = SineNet()

# Convert the mod el to TorchScript through tracing
# This creates an optimized, JIT-compiled version of your model
traced_model = torch.jit.trace(model, x)

# Alternative: Use scripting instead of tracing
# scripted_model = torch.jit.script(model)

optimizer = torch.optim.Adam(traced_model.parameters(), lr=0.0001)

# Training loop with JIT-compiled model
start = time.time()
for epoch in range(4000):
    optimizer.zero_grad()
    
    # Use the traced model instead of the original
    y_pred = traced_model(x)
    
    loss = nn.MSELoss()(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
print(f"Finished training in {time.time()-start} seconds")

# Evaluation
traced_model.eval()

x_test = torch.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_test = torch.sin(x_test)

with torch.no_grad():
    y_pred = traced_model(x_test)

# Plot results
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')

plt.plot(x_test.numpy(), y_test.numpy(), label='True')
plt.plot(x_test.numpy(), y_pred.numpy(), label='Predicted')
plt.legend()
plt.show()
