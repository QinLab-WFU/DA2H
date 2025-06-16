from argparse import Namespace

import torch.nn.functional as F
import torchvision
from torch import nn


def build_model(args: Namespace, pretrained):
    if args.backbone != "resnet50":
        raise NotImplementedError(f"not support: {args.backbone}")
    net = ResNet50Mod(args.tasks, [args.n_bits, args.aux_dim], pretrained)
    return net.cuda(), "cls"


class ResNet50Mod(nn.Module):
    def __init__(self, out_modes, embed_dims, pretrained):
        super().__init__()

        self.out_modes = out_modes

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torchvision.models.resnet50(weights=weights)

        for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.model.modules()):
            module.eval()
            module.train = lambda _: None

        ### Set Embedding Layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.ModuleDict({task: nn.Linear(in_features, embed_dims[i]) for i, task in enumerate(out_modes)})

        ### Resid. Blocks broken down for specific targeting. Primarily used for initial clustering which makes use
        ### of features at different levels.
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        itermasks, out_coll = [], {}

        # Compute First Layer Output
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        if is_init_cluster_generation:
            # If the first clusters before standardization are computed: We use the initial layers with strong
            # average pooling. Using these, we saw much better initial grouping then when using layer combinations or
            # only the last layer.
            x = F.avg_pool2d(self.model.layer1(x), 18, 12)
            x = F.normalize(x.view(x.size(0), -1))
            return x
        else:
            # Run Rest of ResNet
            for block in self.layer_blocks:
                x = block(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)

            # Store the final conv. layer output, it might be useful.
            out_coll["last"] = x
            for out_mode in self.out_modes:
                mod_x = self.model.fc[out_mode](x)
                out_coll[out_mode] = F.normalize(mod_x, dim=-1)
            return out_coll
