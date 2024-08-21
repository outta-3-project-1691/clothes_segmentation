import torch
import torch.nn as nn
import torch.nn.functional as F

class SCLoss(nn.Module):
    def __init__(self):
        super(SCLoss, self).__init__()
    
    def forward(self, pred, target):
        # Pred와 Target은 (B, 1, H, W)의 형태라고 가정
        # 그레이디언트 차이를 계산하기 위해 x와 y 방향으로 편미분
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        target_dx = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_dy = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # x와 y 방향의 그레이디언트 차이를 합산
        loss_x = F.l1_loss(pred_dx, target_dx)
        loss_y = F.l1_loss(pred_dy, target_dy)
        
        return loss_x + loss_y
