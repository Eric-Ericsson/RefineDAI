# --- Configuration ---
PRED_DIR = "result"                  
LABEL_DIR = "dataset/label"
IMAGE_DIR = "dataset/images_low_light" 
                                     
IOU_THRESHOLD = 0.5                  
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
MIN_SSIM_PATCH_DIM = 8

# --- Helper Functions ---

def iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
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
    """Loads detections from a prediction file.
    Format per line: x1 y1 x2 y2 conf
    Returns list of tuples: [(x1, y1, x2, y2, conf)] sorted by confidence desc.
    """
    preds = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    x1, y1, x2, y2, conf = map(float, line.strip().split())
                    
                    if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and conf >= 0:
                         preds.append((x1, y1, x2, y2, conf))
                    else:
                        print(f"Warning: Skipping invalid prediction line in {filepath}: {line.strip()}")
                except ValueError:
                    print(f"Warning: Skipping malformed prediction line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Prediction file not found: {filepath}")
        return []
    
    return sorted(preds, key=lambda x: -x[4])

def load_ground_truth(filepath):
    """Loads ground truth boxes from a label file.
    Format: First line is count N, next N lines are x1 y1 x2 y2 (assuming no class labels needed here)
    Returns list of lists: [[x1, y1, x2, y2], ...]
    """
    gts = []
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            if not lines:
                return []
            try:
                
                potential_count = lines[0].strip()
                start_line = 0
                if potential_count.isdigit() and len(lines) > 1:
                     
                     count = int(potential_count)
                     if count == len(lines) - 1:
                         start_line = 1
                     else:
                         pass

                for i in range(start_line, len(lines)):
                    line = lines[i]
                    try:
                        coords = list(map(float, line.strip().split()[:4]))
                        if len(coords) == 4:
                             
                            if coords[0] >= 0 and coords[1] >= 0 and coords[2] > coords[0] and coords[3] > coords[1]:
                                gts.append(coords)
                            else:
                                print(f"Warning: Skipping invalid ground truth box in {filepath}: {line.strip()}")
                        else:
                             print(f"Warning: Skipping malformed ground truth line (expected 4 coords) in {filepath}: {line.strip()}")
                    except ValueError:
                        print(f"Warning: Skipping non-numeric ground truth line in {filepath}: {line.strip()}")
            except ValueError:
                 print(f"Warning: Could not parse ground truth file (maybe unexpected format?): {filepath}")

    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {filepath}")
       
        return []
    return gts


def get_image_path(base_img_dir, file_id, extensions):
    """Finds an image file matching the file_id with allowed extensions."""
    for ext in extensions:
        img_path = os.path.join(base_img_dir, f"{file_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def extract_patch(image, box):
    """Extracts image patch defined by box [x1, y1, x2, y2].
    Handles boundary conditions and returns None if box is invalid.
    """
    h, w = image.shape[:2]
    
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(w, int(np.ceil(box[2])))
    y2 = min(h, int(np.ceil(box[3])))

    
    if x1 >= x2 or y1 >= y2:
        return None

    patch = image[y1:y2, x1:x2]
    return patch

def calculate_ssim_for_tp(image, pred_box, gt_box):
    """Calculates SSIM between patches defined by pred_box and gt_box from the image.
    Handles potential errors and resizes patches for comparison.
    Returns SSIM score (float) or None if calculation fails.
    """
    pred_patch = extract_patch(image, pred_box)
    gt_patch = extract_patch(image, gt_box)

    if pred_patch is None or gt_patch is None:
        return None

    if pred_patch.shape[0] < MIN_SSIM_PATCH_DIM or pred_patch.shape[1] < MIN_SSIM_PATCH_DIM or \
       gt_patch.shape[0] < MIN_SSIM_PATCH_DIM or gt_patch.shape[1] < MIN_SSIM_PATCH_DIM:
        return None

    if len(pred_patch.shape) > 2 and pred_patch.shape[2] > 1:
        pred_patch_gray = cv2.cvtColor(pred_patch, cv2.COLOR_BGR2GRAY)
    else:
        pred_patch_gray = pred_patch

    if len(gt_patch.shape) > 2 and gt_patch.shape[2] > 1:
        gt_patch_gray = cv2.cvtColor(gt_patch, cv2.COLOR_BGR2GRAY)
    else:
        gt_patch_gray = gt_patch

    target_h, target_w = gt_patch_gray.shape[:2]
    if pred_patch_gray.shape != gt_patch_gray.shape:
        pred_patch_resized = cv2.resize(pred_patch_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        pred_patch_resized = pred_patch_gray 

    # Calculate SSIM
    try:
        data_range = np.max(gt_patch_gray) - np.min(gt_patch_gray)
        if data_range == 0:
             return 1.0 if np.array_equal(pred_patch_resized, gt_patch_gray) else 0.0

        
        win_size = min(7, pred_patch_resized.shape[0], pred_patch_resized.shape[1])
        if win_size % 2 == 0:
            win_size -= 1 
        if win_size < 3:
             return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            score = ssim(pred_patch_resized, gt_patch_gray,
                         data_range=data_range,
                         win_size=win_size,
                         gaussian_weights=True)
            return score
    except ValueError as e:
        print(f"Warning: SSIM calculation failed. Error: {e}. Pred Patch Shape: {pred_patch_resized.shape}, GT Patch Shape: {gt_patch_gray.shape}, win_size: {win_size}. Skipping SSIM for this pair.")
        return None

# --- Main Evaluation Logic ---
print("Starting Evaluation...")
print(f"Prediction Directory: {os.path.abspath(PRED_DIR)}")
print(f"Label Directory:      {os.path.abspath(LABEL_DIR)}")
print(f"Image Directory:      {os.path.abspath(IMAGE_DIR)}")
print(f"IoU Threshold:        {IOU_THRESHOLD}")
print("-" * 30)

# Aggregate metrics counters
total_tp = 0
total_fp = 0
total_fn = 0
all_ssim_scores = []
files_evaluated = 0
files_skipped = 0

pred_files = glob.glob(os.path.join(PRED_DIR, "*.txt"))
if not pred_files:
    print(f"Error: No prediction .txt files found in {PRED_DIR}. Exiting.")
    exit()

print(f"Found {len(pred_files)} prediction files.")

for pred_file_path in pred_files:
    file_id = os.path.splitext(os.path.basename(pred_file_path))[0]

    label_file_path = os.path.join(LABEL_DIR, f"{file_id}.txt")
    image_file_path = get_image_path(IMAGE_DIR, file_id, IMAGE_EXTENSIONS)

    # --- Sanity Checks ---
    if not os.path.exists(label_file_path):
        print(f"Warning: Label file not found for {file_id}. Skipping this file.")
        files_skipped += 1
        continue
    if image_file_path is None:
        print(f"Warning: Image file not found for {file_id} in {IMAGE_DIR}. Skipping this file.")
        files_skipped += 1
        continue

    image = cv2.imread(image_file_path)
    if image is None:
        print(f"Warning: Failed to load image {image_file_path}. Skipping this file.")
        files_skipped += 1
        continue

    predictions = load_predictions(pred_file_path)

    ground_truths = load_ground_truth(label_file_path)

    # --- Evaluate single image ---
    tp = 0
    fp = 0
    image_ssim_scores = []
    matched_gt_indices = set()

    for pred in predictions:
        pred_box = pred[:4]
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt_box in enumerate(ground_truths):
            if i in matched_gt_indices:
                continue

            current_iou = iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i

        if best_iou >= IOU_THRESHOLD:
            if best_gt_idx not in matched_gt_indices:
                tp += 1
                matched_gt_indices.add(best_gt_idx)

                ssim_score = calculate_ssim_for_tp(image, pred_box, ground_truths[best_gt_idx])
                if ssim_score is not None:
                    image_ssim_scores.append(ssim_score)

            else:
                fp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt_indices)

    total_tp += tp
    total_fp += fp
    total_fn += fn
    all_ssim_scores.extend(image_ssim_scores)
    files_evaluated += 1

# --- Final Metrics Calculation ---
print("-" * 30)
print("\nEvaluation Summary:")
print(f"Images evaluated: {files_evaluated}")
if files_skipped > 0:
    print(f"Images skipped (missing labels/images): {files_skipped}")

if files_evaluated == 0:
    print("No images were successfully evaluated.")
else:
    # --- IoU-based metrics ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- IoU-Based Detection Metrics ---")
    print(f"Total True Positives (TP):  {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # --- SSIM metric ---
    print("\n--- SSIM Metric (Visual Similarity for TPs) ---")
    if all_ssim_scores:
        average_ssim = np.mean(all_ssim_scores)
        std_dev_ssim = np.std(all_ssim_scores)
        print(f"Average SSIM over {len(all_ssim_scores)} True Positive detections: {average_ssim:.4f}")
        print(f"Standard Deviation of SSIM: {std_dev_ssim:.4f}")
       
    elif total_tp > 0:
         print("Average SSIM: N/A (SSIM calculation failed or patches too small for all TPs)")
    else:
        print("Average SSIM: N/A (No True Positives found)")

print("-" * 30)
print("Evaluation Complete.")