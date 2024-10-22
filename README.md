# ONNX DApp Template

This is a template for PyTorch Cartesi DApps in python. It uses python3 to execute the backend application.
The application entrypoint is the `dapp.py` file.

## How to use it
1. Replace the simple_nn_model.pth file by the model you've trained.
2. Ensure that the input shape matches the model requirements.

The placeholder model code as an example of exporting using TorchScript:
```python
import torch
import torch.nn as nn

# Define a simple model class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize and train your model (omitting training code for brevity)
model = SimpleNN()

# Dummy input for tracing
dummy_input = torch.randn(1, 10)

# Trace the model with dummy input
traced_model = torch.jit.trace(model, dummy_input)

# Save the traced model
traced_model.save('simple_nn_model.pth')

print("Model saved with TorchScript.")

```
