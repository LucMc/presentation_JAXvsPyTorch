import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax.random import PRNGKey
import time


# Define the model
class SineModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        net = nn.Dense(32)(x)
        net = nn.relu(net)
        net = nn.Dense(32)(net)
        net = nn.relu(net)
        net = nn.Dense(1)(net)
        return net


# Prepare the data
x = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 200).reshape(-1, 1)
y = jnp.sin(x)

# Create a model and initialize it
model = SineModel()
params = model.init(PRNGKey(0), x)


# Define the loss function
def compute_loss(params, x, y):
    y_pred = model.apply(params, x)
    return jnp.mean(jnp.square(y_pred - y))


# Define the optimizer
optimizer = optax.adam(0.0003)

# Initialize the optimizer state.
opt_state = optimizer.init(params)


# Training loop
@jax.jit
def update(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(compute_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


# Training loop
start = time.time()
for epoch in range(4000):
    params, opt_state, loss = update(params, opt_state, x, y)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print(f"Finished training in {time.time()-start} seconds")
# Check trained model

y_pred = model.apply(params, x)

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use(
    "QtAgg"
)  # Because I'm running from terminal, change to whatever backend you want

# Generate a fine grid of inputs for visualization
# x_test is the same as x but feel free to change and investigate
x_test = jnp.linspace(-2 * jnp.pi, 2 * jnp.pi, 1000).reshape(-1, 1)

# Compute the network's output on this grid
y_test_pred = model.apply(params, x_test)

plt.figure(figsize=(10, 5))
plt.plot(x, y, "b-", label="true")
plt.plot(x_test, y_test_pred, "r-", label="predicted")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("True and predicted function values")
plt.grid(True)
plt.show()
