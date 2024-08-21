import torch
from torch.functional import F

def calculate_f1(preds, targets, num_classes):
    epsilon = 1e-7
    total_tp = torch.zeros(num_classes)
    total_fp = torch.zeros(num_classes)
    total_fn = torch.zeros(num_classes)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)

        # True Positives, False Positives, False Negatives 계산
        tp = (pred_inds & target_inds).sum().float().item()
        fp = (pred_inds & ~target_inds).sum().float().item()
        fn = (~pred_inds & target_inds).sum().float().item()

        total_tp[cls] += tp
        total_fp[cls] += fp
        total_fn[cls] += fn

    # 각 클래스별 Precision, Recall, F1 스코어 계산
    f1_scores = []
    for cls in range(num_classes):
        precision = total_tp[cls] / (total_tp[cls] + total_fp[cls] + epsilon)
        recall = total_tp[cls] / (total_tp[cls] + total_fn[cls] + epsilon)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    # 전체 평균 F1 스코어 반환
    return torch.mean(torch.tensor(f1_scores)).item()

def evaluate_F1(model, dataloader, num_classes=7, device='cpu'):
    model.eval()
    total_f1_score = 0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            # 모델 예측
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # outputs: [b, 7, 128, 128]
            
            # 소프트맥스 후 argmax로 예측된 클래스 선택
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # preds: [b, 128, 128]

            # F1 스코어 계산
            batch_f1_score = calculate_f1(preds, targets, num_classes)
            total_f1_score += batch_f1_score
            total_batches += 1

    # 전체 배치에 대한 평균 F1 스코어 계산
    mean_f1_score = total_f1_score / total_batches
    return mean_f1_score