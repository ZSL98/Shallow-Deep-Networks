import torch
import torch.nn as nn
from networks import custom_resnet_p1, custom_resnet_p2
import torchvision.models as models

dummy_input = torch.randn(1, 3, 224, 224)
model_p1 = custom_resnet_p1()

input_names = ["input"]
output_names = ["output1", "output2"]
torch.onnx.export(model_p1, dummy_input, "custom_resnet_p1.onnx",input_names=input_names, output_names=output_names,
                  verbose=True,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'output1': {0: 'batch_size'},
                                      'output2': {0: 'batch_size'},
                                  },opset_version=11)