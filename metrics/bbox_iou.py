import torch
from torchvision.ops import box_iou


def bbox_iou(pred_boxes: torch.tensor, target_boxes: torch.tensor):

    # normalized box value is too small, so that the area is 0.
    ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
    ious = torch.einsum('i i -> i', ious)  # take diag elem
    miou = ious.mean().item()
    correct = (ious > 0.5).sum().item()
    num = len(target_boxes)

    return {
        'accuracy': 1.0 * correct / num,
        'miou': miou,
        "num": num,
    }


if __name__ == '__main__':
    box1 = torch.tensor([[2, 3.1, 7, 5], [3, 4, 8, 4.8], [4, 4, 5.6, 7]])
    box2 = torch.tensor([[2, 4, 7, 5], [3, 4, 8, 4.8], [4, 4, 5.6, 7]])

    res = bbox_iou(box1, box2)
    print(res)

