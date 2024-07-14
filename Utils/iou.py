import torch

def intersection_over_union(boxes_preds:torch.tensor, boxes_labels:torch.tensor, box_format="midpoint"):
    """
    Calculate intersection over union (IoU) for bounding boxes.

    Args:
        boxes_preds (torch.Tensor): Predicted bounding boxes of shape (N, 4), where N is the batch size.
        boxes_labels (torch.Tensor): Ground truth bounding boxes of shape (N, 4).
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        torch.Tensor: IoU values of shape (N,).
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    else:
        raise ValueError("Invalid box format. Use 'midpoint' or 'corners'.")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection + 1e-10
    iou = intersection / union
    return iou

# Test Cases
def run_test_cases():
    # Test Case 1: Perfect Overlap
    boxes_preds = torch.tensor([[0, 0, 2, 2]])
    boxes_labels = torch.tensor([[0, 0, 2, 2]])
    box_format = "midpoint"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 1 IoU: {iou}")  # Expected output: tensor([1.])

    # Test Case 2: No Overlap
    boxes_preds = torch.tensor([[0, 0, 2, 2]])
    boxes_labels = torch.tensor([[5, 5, 2, 2]])
    box_format = "midpoint"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 2 IoU: {iou}")  # Expected output: tensor([0.])

    # Test Case 3: Partial Overlap
    boxes_preds = torch.tensor([[0, 0, 4, 4]])
    boxes_labels = torch.tensor([[1, 1, 4, 4]])
    box_format = "midpoint"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 3 IoU: {iou}")  # Expected output: value between 0 and 1

    # Test Case 4: Different Sizes, Partial Overlap
    boxes_preds = torch.tensor([[1, 1, 2, 2]])
    boxes_labels = torch.tensor([[0, 0, 4, 4]])
    box_format = "midpoint"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 4 IoU: {iou}")  # Expected output: value between 0 and 1

    # Test Case 5: Boxes with Corners Format, Perfect Overlap
    boxes_preds = torch.tensor([[0, 0, 2, 2]])
    boxes_labels = torch.tensor([[0, 0, 2, 2]])
    box_format = "corners"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 5 IoU: {iou}")  # Expected output: tensor([1.])

    # Test Case 6: Boxes with Corners Format, No Overlap
    boxes_preds = torch.tensor([[0, 0, 2, 2]])
    boxes_labels = torch.tensor([[3, 3, 5, 5]])
    box_format = "corners"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 6 IoU: {iou}")  # Expected output: tensor([0.])

    # Test Case 7: Multiple Boxes, Mixed Overlaps
    boxes_preds = torch.tensor([[0, 0, 2, 2], [0, 0, 3, 3], [1, 1, 2, 2]])
    boxes_labels = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 3, 3]])
    box_format = "midpoint"
    iou = intersection_over_union(boxes_preds, boxes_labels, box_format)
    print(f"Test Case 7 IoU: {iou}")  # Expected output: tensor([1., 0.1429, 0.])

# Run test cases
run_test_cases()
