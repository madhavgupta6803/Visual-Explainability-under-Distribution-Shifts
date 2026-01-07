import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import supervision as sv
from sklearn import metrics
import argparse
import cv2.ximgproc 
import re
plt.rc('font', family="Arial")

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from interpretation.submodular_detection import DetectionSubModularExplanation
from tqdm import tqdm
import glob

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Visualization functions
def add_value(S_set, json_file):
    single_mask = np.zeros_like(S_set[0], dtype=np.float16)
    value_list_1 = np.array(json_file["smdl_score"])
    value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
    value_list = value_list_1 - value_list_2
    
    values = []
    value = 0
    for smdl_single_mask, smdl_value in zip(S_set, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask==1] = value
        values.append(value)
    
    attribution_map = single_mask - single_mask.min()
    attribution_map = attribution_map / (attribution_map.max() + 1e-6)
    return attribution_map, np.array(values)

def gen_cam(image_path, mask):
    w = mask.shape[1]
    h = mask.shape[0]
    image = cv2.resize(cv2.imread(image_path), (w, h))
    mask = cv2.resize(mask, (int(w/20), int(h/20)))
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
    heatmap = np.float32(heatmap)
    cam = 0.5 * heatmap + 0.5 * np.float32(image)
    return cam.astype(np.uint8), heatmap.astype(np.uint8)

def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= (np.max(image) + 1e-6)
    image *= 255.
    return np.uint8(image)

def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    class_ids = np.zeros(len(boxes), dtype=int)
    h, w, _ = image.shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
    bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
    label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
    annotated_frame = image
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
    return annotated_frame

def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
    S_set_add = S_set.copy()
    S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
    image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
    if mode == "insertion":
        curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
    elif mode == "deletion":
        curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

    if index is None:
        ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
    else:
        ours_best_index = index
    
    x = [0.0] + saved_json_file["region_area"]
    i = len(x)
    
    fig = plt.figure(figsize=(30, 8))
    ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
    ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
    ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title('Attribution Map', fontsize=54)
    ax1.set_facecolor('white')
    ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.xaxis.set_visible(True)
    ax2.yaxis.set_visible(False)
    ax2.set_title('Searched Region', fontsize=54)
    ax2.set_facecolor('white')
    ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    yticks = ax3.get_yticks()
    yticks = yticks[yticks != 0]
    ax3.set_yticks(yticks)
    ax3.set_ylabel('Object Score', fontsize=44)
    ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
    ax3.tick_params(axis='both', which='major', labelsize=36)

    curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
    x_ = x[:i]
    ours_y = curve_score[:i]
    ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)
    ax3.set_facecolor('white')
    ax3.spines['bottom'].set_color('black')
    ax3.spines['bottom'].set_linewidth(2.0)
    ax3.spines['top'].set_color('none')
    ax3.spines['left'].set_color('black')
    ax3.spines['left'].set_linewidth(2.0)
    ax3.spines['right'].set_color('none')
    ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
    ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
    ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

    kernel = np.ones((10, 10), dtype=np.uint8)
    if mode == "insertion":
        mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
    elif mode == "deletion":
        mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

    if ours_best_index != 0:
        dilate = cv2.dilate(mask, kernel, iterations=3)
        edge = dilate - mask
    else:
        edge = np.zeros_like(mask)

    image_debug = image_baseline.copy()
    image_debug[mask > 0] = image_debug[mask > 0] * 0.3
    if ours_best_index != 0:
        image_debug[edge > 0] = np.array([0, 0, 255])

    if mode == "insertion":
        if ours_best_index != 0:
            target_box = saved_json_file["insertion_box"][ours_best_index - 1]
            cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
        else:
            target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
            cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
        color = (255, 69, 0)
    elif mode == "deletion":
        if ours_best_index != 0:
            target_box = saved_json_file["deletion_box"][ours_best_index - 1]
            cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
        else:
            target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
            cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
        color = (30, 144, 255)

    image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
    image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
    ax2.imshow(image_debug[..., ::-1])

    auc = metrics.auc(x, curve_score)
    ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} {auc:.4f}", fontsize=54)
    
    return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model on iNaturalist Dataset')
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/INaturalist',
                        help='Path to iNaturalist dataset directory')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/INaturalist_correct_Detection_grounding.json',
                        help='Path to detection JSON file')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="Superpixel algorithm")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='Lambda1 for submodular explanation')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='Lambda2 for submodular explanation')
    parser.add_argument('--division-number', 
                        type=int, default=50,
                        help='Number of superpixel regions')
    parser.add_argument('--num-images', 
                        type=int, default=100,
                        help='Number of images to process')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/grounding-dino-inaturalist/',
                        help='Output directory for results')
    parser.add_argument('--device', 
                        type=str, default="cuda:2",
                        help='Device to run model (cuda:2)')
    return parser.parse_args()

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def normalize_class_name(class_name: str) -> str:
    """Normalize class names, ensuring non-empty output."""
    norm_name = re.sub(r'[^a-zA-Z0-9\s]', '', class_name.lower().strip())
    return norm_name if norm_name else class_name.lower().replace(' ', '_')

def SubRegionDivision(image, mode="slico", region_size=30):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
        slic.iterate(20)
        label_slic = slic.getLabels()
        number_slic = slic.getNumberOfSuperpixels()
        print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
        for i in range(number_slic):
            img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
            if img_copp.shape[:2] != image.shape[:2]:
                raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image, 10)
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()
        print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
        for i in range(number_seeds):
            img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
            if img_copp.shape[:2] != image.shape[:2]:
                raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
            element_sets_V.append(img_copp)
    return element_sets_V

def transform_vision_data(image, device='cuda:2'):
    image = Image.fromarray(image)
    image_transformed, _ = data_transform(image, None)
    image_transformed = image_transformed.to(device)
    return image_transformed

def convert_bbox_to_xyxy(bbox, image_shape):
    """Convert [x, y, w, h] to [x1, y1, x2, y2] and clamp to image boundaries."""
    x, y, w, h = bbox
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    img_h, img_w = image_shape[:2]
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return [x1, y1, x2, y2]

def find_image_path(img_folder, filename):
    """Helper function to find the correct image path for iNaturalist dataset"""
    base_filename = os.path.basename(filename)
    
    # Strategy 1: Try original path from JSON
    img_path = os.path.join(img_folder, filename)
    if os.path.exists(img_path):
        return img_path
    
    # Strategy 2: Try without 'val/' prefix
    img_path_alt = os.path.join(img_folder, filename.replace('val/', '', 1))
    if os.path.exists(img_path_alt):
        return img_path_alt
    
    # Strategy 3: Search through all category directories
    if os.path.exists(img_folder):
        for dir_name in os.listdir(img_folder):
            dir_path = os.path.join(img_folder, dir_name)
            if os.path.isdir(dir_path):
                potential_path = os.path.join(dir_path, base_filename)
                if os.path.exists(potential_path):
                    return potential_path
    
    # Strategy 4: Try just the basename in the main folder
    img_path_base = os.path.join(img_folder, base_filename)
    if os.path.exists(img_path_base):
        return img_path_base
    
    return None

class GroundingDino_Adaptation(torch.nn.Module):
    def __init__(self, detection_model, device="cuda:2"):
        super().__init__()
        self.detection_model = detection_model.to(device)
        self.device = device
        self.caption = None
    
    def forward(self, images, h, w):
        batch = images.shape[0]
        captions = [self.caption for _ in range(batch)]
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("Input images contain NaN or Inf values")
        images = images.to(self.device)
        with torch.no_grad():
            outputs = self.detection_model(images, captions=captions)
        prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
        prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
        positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
        if positive_map is not None:
            print(f"positive_map shape: {positive_map.shape}")
        boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return xyxy, prediction_logits, positive_map

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_class_mapping(eval_list_data):
    """Build a mapping from category names to indices for the text prompt"""
    categories = set()
    for info in eval_list_data["case1"]:
        if "category" in info:
            categories.add(info["category"])
    
    # Sort categories for consistent ordering
    sorted_categories = sorted(list(categories))
    
    # Normalize category names for text prompt
    normalized_categories = [normalize_class_name(cat) for cat in sorted_categories]
    
    # Create text prompt
    text_prompt = " . ".join(normalized_categories) + " ."
    
    # Create mapping from original category names to indices
    category_to_index = {cat: idx for idx, cat in enumerate(sorted_categories)}
    
    print(f"Found {len(sorted_categories)} unique categories")
    print(f"Sample categories: {sorted_categories[:10]}")
    print(f"Text prompt length: {len(text_prompt)} characters")
    
    return category_to_index, text_prompt, sorted_categories

def main(args):
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    
    model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
    detection_model = GroundingDino_Adaptation(model, device=args.device)
    
    # Load evaluation data
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    print("First 5 JSON entries:")
    for info in val_file["case1"][:5]:
        print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Bbox: {info.get('bbox')}")
    
    # Build category mapping and text prompt
    category_to_index, text_prompt, sorted_categories = build_class_mapping(val_file)
    caption = preprocess_caption(text_prompt)
    detection_model.caption = caption
    print(f"Using text prompt with {len(sorted_categories)} categories")
    
    smdl = DetectionSubModularExplanation(
        detection_model,
        lambda x: transform_vision_data(x, device=args.device),
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=args.device,
        batch_size=4  # Reduced to avoid OOM
    )
    
    # Filter available images
    available_images = []
    for info in val_file["case1"]:
        filename = info["file_name"]
        image_path = find_image_path(args.Datasets, filename)
        if image_path is not None:
            available_images.append(info)
        else:
            print(f"Warning: Could not find image file for {filename}")
    
    print(f"Found {len(available_images)} available images out of {len(val_file['case1'])} total")
    
    if not available_images:
        print(f"Error: No images found in {args.Datasets}")
        return
    
    # Select subset of images to process
    num_images = min(args.num_images, len(available_images))
    if num_images == 0:
        print(f"Error: No valid images found")
        return
    
    selected_images = np.random.choice(len(available_images), size=num_images, replace=False)
    select_infos = [available_images[i] for i in selected_images]
    print(f"Selected {len(select_infos)} images for processing")
    
    # Create output directories
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
        args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
    mkdir(save_dir)
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    save_vis_root_path = os.path.join(save_dir, "visualization")
    mkdir(save_vis_root_path)
    
    id = 1
    for info in tqdm(select_infos, desc="Processing images"):
        filename = info["file_name"]
        save_json_path = os.path.join(save_json_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.json")
        save_npy_path = os.path.join(save_npy_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.npy")
        save_vis_path = os.path.join(save_vis_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.png")
        
        # Skip if already processed
        if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
            id += 1
            continue

        if "category" not in info:
            print(f"Warning: Category not found in JSON for {filename}, skipping")
            continue
        
        category = info["category"]
        if category not in category_to_index:
            print(f"Warning: Category {category} not found in category mapping for {filename}, skipping")
            continue
        
        target_class = category_to_index[category]
        
        if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
            print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
            continue
        
        # Find image path
        image_path = find_image_path(args.Datasets, filename)
        if image_path is None:
            print(f"Failed to find image: {filename}")
            continue
            
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
        target_box = convert_bbox_to_xyxy(info["bbox"], image.shape)
        
        class_name = info.get("category", "unknown")
        
        torch.cuda.empty_cache()
        
        image_proccess = transform_vision_data(image, device=args.device)
        image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
        if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
            print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
            continue
        
        region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
        try:
            V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
        except ValueError as e:
            print(f"Error in SubRegionDivision for {filename}: {e}")
            continue
        
        try:
            S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
        except Exception as e:
            print(f"Error in submodular explanation for {filename}: {e}")
            continue
        
        # Save npy and json
        np.save(save_npy_path, np.array(S_set))
        with open(save_json_path, "w") as f:
            json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
        # Generate and save visualization
        try:
            attribution_map, _ = add_value(S_set, saved_json_file)
            vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
            vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
            vis_saliency_map_w_box = annotate_with_grounding_dino(
                vis_saliency_map,
                np.array([saved_json_file["target_box"]]),
                [f"{class_name}: {saved_json_file['insertion_cls'][-1] if saved_json_file['insertion_cls'] else 0:.2f}"]
            )
            fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
            fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"Saved visualization for {filename} at {save_vis_path}")
        except Exception as e:
            print(f"Error in visualization for {filename}: {e}")
            continue
        
        id += 1
        torch.cuda.empty_cache()  # Clear after each image
    
    print(f"Processed {id-1} images, results saved in {save_dir}")
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)