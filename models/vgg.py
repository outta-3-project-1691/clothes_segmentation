import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.models.vgg as vgg

# 모델의 층을 초기화 시킬 때 쓴 코드로, 필요하지 않으시다면 사용하지 않으셔도 됩니다.
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]

    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


# 1. VGGNet 기반 segmentation    
class segmentation_model(nn.Module):
    def __init__(self, n_class=7, device=None):
        super(segmentation_model, self).__init__()
        # [1] 빈칸을 작성하시오.

        # VGG16 모델 불러오기
        vgg_model = models.vgg16(pretrained=True)
        features = list(vgg_model.features.children())
        
        # 필요한 feature layers를 가져옵니다.
        self.layer1 = nn.Sequential(*features[:5])  # Conv1
        self.layer2 = nn.Sequential(*features[5:10])  # Conv2
        self.layer3 = nn.Sequential(*features[10:17])  # Conv3
        self.layer4 = nn.Sequential(*features[17:24])  # Conv4
        self.layer5 = nn.Sequential(*features[24:])  # Conv5

        # Fully Convolutional layers
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Classifying layer
        self.score = nn.Conv2d(4096, n_class, kernel_size=1)
        
        # Deconvolution layers
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1)
        self.upscore4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4)
        
        # Auxiliary layers to fuse the lower layers
        self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)
        
        self._initialize_weights()
        self.device = device

    def _initialize_weights(self):
        # [2] 빈칸을 작성하시오.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # [3] 빈칸을 작성하시오.
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        x6 = self.relu6(self.conv6(x5))
        x6 = self.drop6(x6)
        x7 = self.relu7(self.conv7(x6))
        x7 = self.drop7(x7)
        score = self.score(x7)
        
        upscore2 = self.upscore2(score)
        upscore2 = self.upscore4(upscore2)

        score_pool4 = self.score_pool4(x4)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore2(fuse_pool4)
        
        score_pool3 = self.score_pool3(x3)
        fuse_pool3 = upscore_pool4 + score_pool3
        
        out = self.upscore8(fuse_pool3)
        
        return out