import os
import glob
import numpy as np

# Paths
pred_dir = "result"
label_dir = "dataset/label"

def iou(box1, box2):
    """Compute IoU of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def load_predictions(filepath):
    preds = []
    with open(filepath, "r") as f:
        for line in f:
            x1, y1, x2, y2, conf = map(float, line.strip().split())
            preds.append((x1, y1, x2, y2, conf))
    return sorted(preds, key=lambda x: -x[4])  # Sort by confidence

def load_ground_truth(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        if not lines:
            return []
        count = int(lines[0].strip())
        return [list(map(float, line.strip().split())) for line in lines[1:count+1]]

# Metrics counters
total_tp = total_fp = total_fn = 0
files_evaluated = 0

# Loop through all prediction files
for pred_file in glob.glob(os.path.join(pred_dir, "*.txt")):
    file_id = os.path.splitext(os.path.basename(pred_file))[0]
    label_file = os.path.join(label_dir, f"{file_id}.txt")

    if not os.path.exists(label_file):
        print(f"Skipping {file_id}: label file not found.")
        continue

    preds = load_predictions(pred_file)
    gts = load_ground_truth(label_file)
    matched_gt = set()

    tp = 0
    fp = 0

    for pred in preds:
        pred_box = pred[:4]
        match_found = False
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            if iou(pred_box, gt) >= 0.5:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(gts) - tp

    total_tp += tp
    total_fp += fp
    total_fn += fn
    files_evaluated += 1

# Final metrics
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\nEvaluation Summary:")
print(f"Images evaluated: {files_evaluated}")
print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
