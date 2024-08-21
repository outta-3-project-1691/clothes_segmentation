import torch
from torch.functional import F

# preds, labels -> torch.Tensor
def calculate_miou(preds, targets, num_classes=7):
    ious = []
    for i in range(num_classes):
        tp = torch.sum((preds == i) & (targets == i)).float()
        fp = torch.sum((preds == i) & (targets != i)).float()
        fn = torch.sum((preds != i) & (targets == i)).float()

        # IoU = Intersection (TP) / Union (TP + FP + FN)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else torch.tensor(0.0).to('cuda')
        ious.append(iou.item())
    
    return torch.mean(torch.tensor(ious)).item()

def evaluate_mIoU(model, dataloader, num_classes=7, device='cpu'):
    model.eval()
    total_mIoU_score = 0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            
            # 모델 예측
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # outputs: [b, 7, 128, 128]
            
            # 소프트맥스 후 argmax로 예측된 클래스 선택
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # preds: [b, 128, 128]

            # F1 스코어 계산
            mIoU_score = calculate_miou(preds, targets, num_classes)
            total_mIoU_score += mIoU_score
            total_batches += 1
    
    mean_mIoU_score = total_mIoU_score / total_batches

    return mean_mIoU_score
