import torch


def evaluate(model, loader, loss_fn, device):
    """(정확도, 평균 손실)"""
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            total_loss += loss_fn(scores, labels).item() * len(labels)
            correct += (scores.argmax(1) == labels).sum().item()
    model.train()
    n = len(loader.dataset)
    return correct / n, total_loss / n
