import torch
import torch.nn.functional as F

from torchvision.models import vgg16


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg16(pretrained=True).features[0:16]
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dark, gt):
        loss = []
        dark_features = self.output_features(dark)
        gt_features = self.output_features(gt)
        for dark_feature, gt_feature in zip(dark_features, gt_features):
            loss.append(F.mse_loss(dark_feature, gt_feature))

        return sum(loss) / len(loss)
