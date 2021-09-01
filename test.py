import torch
import torchvision.models as models

features_in_hook = []
features_out_hook = []

def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

model = models.resnet18(pretrained=True, progress=True)
layer_name = 'layer4.0.downsample.1'
for (name, module) in model.named_modules():
    print(name)
    if name == layer_name:
        module.register_forward_hook(hook=hook)

x  = torch.randn([1, 3, 224, 224])
out = model(x)
print(features_in_hook[0][0].shape)