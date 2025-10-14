import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # First linear: 2 -> 5 (no bias)
        self.linear1 = nn.Linear(2, 5, bias=False)
        # Mul parameter: (1,5) so it matches output of linear1
        self.mul_param = nn.Parameter(torch.randn(1, 5))
        # Second linear: 5 -> 7 (no bias)
        self.linear2 = nn.Linear(5, 7, bias=False)

    def forward(self, x):
        x = self.linear1(x)        # (2,2) -> (2,5)
        x = x * self.mul_param     # elementwise mul (broadcast)
        x = self.linear2(x)        # (2,5) -> (2,7)
        return x

# Create model & input
model = CustomModel()
model.eval()
inp = torch.randn(2, 2)  # fixed input shape

# Export ONNX with fixed sizes (no dynamic axes)
torch.onnx.export(
    model,
    inp,
    "../simple_model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17
)

print("Model exported to custom_model_fixed.onnx (fixed shape 2x2 → 2x7)")