import torch
from collections import Counter
from iou import intersection_over_union

def calculate_iou_for_class(detections, ground_truths, box_format, iou_threshold):
    amount_bboxes = Counter([gt[0] for gt in ground_truths])
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)

    if total_true_bboxes == 0:
        return TP, FP, 0

    for detection_idx, detection in enumerate(detections):
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        else:
            FP[detection_idx] = 1

    return TP, FP, total_true_bboxes

def compute_precision_recall(TP, FP, total_true_bboxes, epsilon=1e-6):
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    return precisions, recalls

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [detection for detection in pred_boxes if detection[1] == c]
        ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

        TP, FP, total_true_bboxes = calculate_iou_for_class(
            detections, ground_truths, box_format, iou_threshold
        )

        if total_true_bboxes == 0:
            continue

        precisions, recalls = compute_precision_recall(TP, FP, total_true_bboxes, epsilon)
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions) , average_precisions


# if __name__ == "__main__":
#     pred_boxes = [
#         [0, 0, 0.9, 0.5, 0.5, 1.0, 1.0],
#         [0, 0, 0.8, 0.4, 0.4, 0.9, 0.9],
#         [1, 1, 0.75, 0.3, 0.3, 0.7, 0.7],
#         [1, 1, 0.6, 0.2, 0.2, 0.6, 0.6]
#     ]
#     true_boxes = [
#         [0, 0, 1.0, 0.5, 0.5, 1.0, 1.0],
#         [1, 1, 1.0, 0.3, 0.3, 0.7, 0.7]
#     ]
#     iou_threshold = 0.5
#     box_format = "midpoint"
#     num_classes = 2

#     mAP, L = mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, num_classes)
#     print(f"Mean Average Precision: {mAP}")
#     print(L)