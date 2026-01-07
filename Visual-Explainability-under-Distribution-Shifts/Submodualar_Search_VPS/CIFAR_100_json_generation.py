"""
Created on 2025/09/10

@author: Adapted from Ruoyu Chen
Generate CIFAR100_correct_Detection_grounding.json with predicted bounding boxes for CIFAR-100 dataset
"""

import os
import json
import numpy as np
import torch
import supervision as sv
from PIL import Image
from tqdm import tqdm
import argparse
import cv2
import groundingdino.datasets.transforms as T
from groundingdino.util import get_tokenlizer
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from torch.utils.data import DataLoader, Dataset
import tensorflow.keras.datasets.cifar100 as cifar100
from torchvision.ops import box_convert
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Data transform for small CIFAR-100 images
data_transform = T.Compose([
    T.RandomResize([64], max_size=64),  # Minimal resizing for 32x32 images
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CIFAR100_correct_Detection_grounding.json with predicted bounding boxes")
    parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config file")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint file")
    parser.add_argument("--image_dir", type=str, default="datasets/CIFAR100", help="CIFAR-100 image directory")
    parser.add_argument("--output_path", type=str, default="datasets/CIFAR100_correct_Detection_grounding.json", help="Output JSON file path")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Confidence threshold for predictions")
    parser.add_argument("--nms_threshold", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run model (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--max_categories", type=int, default=100, help="Maximum number of categories for text prompt")
    parser.add_argument("--batch_processing", action="store_true", help="Enable batch processing to handle large category lists")
    parser.add_argument("--categories_per_batch", type=int, default=5, help="Number of categories per batch when batch processing is enabled")
    return parser.parse_args()

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result

def save_cifar100_images(image_dir, num_images):
    """Save CIFAR-100 validation images to class-based folders and create COCO-style JSON"""
    # Load CIFAR-100 test set
    log.info("Starting CIFAR-100 dataset download/loading...")
    (_, _), (x_test, y_test) = cifar100.load_data()
    log.info("CIFAR-100 dataset loaded successfully.")
    classes = CIFAR100_CLASSES
    log.info(f"Loaded CIFAR-100 with {len(classes)} classes: {classes[:10]}...")

    # Create COCO-style JSON
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    # Create directories for each class
    os.makedirs(image_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(image_dir, class_name), exist_ok=True)

    # Save images and build JSON with tqdm progress bar
    num_images = min(num_images, len(x_test))
    for idx in tqdm(range(num_images), desc="Saving CIFAR-100 images"):
        img_array = x_test[idx]  # Shape: (32, 32, 3), uint8, RGB
        label = int(y_test[idx, 0])  # Shape: (10000, 1) -> scalar
        class_name = classes[label]
        image_id = idx + 1  # 1-based IDs
        file_name = f"{class_name}/{image_id:05d}.png"
        file_path = os.path.join(image_dir, file_name)

        # Convert NumPy array to PIL Image
        img = Image.fromarray(img_array)
        img.save(file_path)

        # Add to COCO JSON
        coco_json["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": 32,
            "height": 32
        })
        coco_json["annotations"].append({
            "image_id": image_id,
            "category_id": label,
            "bbox": [4, 4, 24, 24]  # Centered 80% of 32x32 image
        })

    # Save COCO JSON
    coco_json_path = os.path.join(image_dir, "val.json")
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f, indent=4, ensure_ascii=False)
    log.info(f"Saved {num_images} CIFAR-100 images to {image_dir} and COCO JSON to {coco_json_path}")

    return coco_json

def safe_collate_fn(batch):
    """Custom collate function that handles None values"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None, None
    from groundingdino.util.misc import collate_fn
    return collate_fn(batch)

def normalize_class_name(class_name: str) -> str:
    """Normalize class names, preserving underscores."""
    norm_name = class_name.lower().strip()
    return norm_name if norm_name else class_name.lower()

def estimate_token_count(cat_list, tokenizer):
    """Estimate token count for a list of categories"""
    normalized_names = [normalize_class_name(cat) for cat in cat_list]
    text_prompt = " . ".join(normalized_names) + " ."
    caption = preprocess_caption(text_prompt)
    tokenized = tokenizer(caption)
    return len(tokenized['input_ids'])

def split_categories_by_token_limit(cat_list, tokenizer, max_tokens=400):
    """Split categories into batches that fit within token limit"""
    batches = []
    current_batch = []
    for cat in cat_list:
        test_batch = current_batch + [cat]
        token_count = estimate_token_count(test_batch, tokenizer)
        if token_count > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [cat]
        else:
            current_batch.append(cat)
    if current_batch:
        batches.append(current_batch)
    return batches

def find_image_path(img_folder, img_info):
    """Helper function to find the correct image path for CIFAR-100 dataset"""
    file_path = os.path.join(img_folder, img_info['file_name'])
    if os.path.exists(file_path):
        return file_path
    log.warning(f"Image not found: {file_path}")
    return None

def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    tokenizer = get_tokenlizer.get_tokenlizer(args.text_encoder_type if hasattr(args, 'text_encoder_type') else "bert-base-uncased")
    return model.to(device), tokenizer

class CIFAR100Detection(Dataset):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__()
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.img_folder = img_folder
        self.transforms = transforms
        self.ids = [img['id'] for img in self.coco['images']]
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        self.cat_list = [cat['name'] for cat in self.coco['categories']]
        log.info(f"Loaded {len(self.cat_list)} categories, e.g., {self.cat_list[:5]}")

        # Test image path resolution
        self._test_image_paths()

    def _test_image_paths(self):
        """Test and log image path resolution"""
        if not self.coco['images']:
            log.warning("No images found in COCO JSON to test path resolution.")
            return
        sample_img = self.coco['images'][0]
        log.info(f"Testing image path resolution for: {sample_img['file_name']}")
        found_path = find_image_path(self.img_folder, sample_img)
        if found_path:
            log.info(f"Successfully resolved image path: {found_path}")
        else:
            log.error(f"Could not resolve path for sample image: {sample_img['file_name']}")

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        img_path = find_image_path(self.img_folder, img_info)
        if img_path is None:
            log.error(f"Could not find image file for {img_info['file_name']}")
            return None, None, img_id
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            log.error(f"Failed to load image {img_path}: {e}")
            return None, None, img_id
        w, h = img.size
        anns = [ann for ann in self.coco['annotations'] if ann['image_id'] == img_id]
        if not anns:
            log.warning(f"No annotations for image ID {img_id}")
            return None, None, img_id
        category_ids = torch.as_tensor([ann['category_id'] for ann in anns])
        target = {
            'image_id': img_id,
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'category_ids': category_ids
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, img_id

    def __len__(self):
        return len(self.ids)

class PostProcessCIFAR100Grounding(torch.nn.Module):
    def __init__(self, num_select=50, cat_list=None, tokenizer=None, num_queries=900):
        super().__init__()
        self.num_select = num_select
        self.num_queries = num_queries
        assert cat_list is not None
        normalized_cat_list = [normalize_class_name(cat) for cat in cat_list]
        captions, cat2tokenspan = build_captions_and_token_span(normalized_cat_list, True)
        # log.info(f"Captions: {captions[:100]}...")
        # log.info(f"Token spans sample: {list(cat2tokenspan.items())[:5]}")
        # log.info(f"Valid categories: {len(normalized_cat_list)}, Sample: {normalized_cat_list[:5]}")
        tokenized = tokenizer(captions)
        
        tokenspanlist = []
        filtered_cat_list = []
        for i, cat in enumerate(normalized_cat_list):
            if cat in cat2tokenspan:
                span = cat2tokenspan[cat]
                # Normalize: take the first valid pair
                if isinstance(span, list):
                    if isinstance(span[0], list) or isinstance(span[0], tuple):
                        s, e = span[0]
                        tokenspanlist.append([(s, e)])
                        filtered_cat_list.append(cat_list[i])
                    elif len(span) == 2 and all(isinstance(x, int) for x in span):
                        s, e = span
                        tokenspanlist.append([(s, e)])
                        filtered_cat_list.append(cat_list[i])
                elif isinstance(span, tuple) and len(span) == 2:
                    tokenspanlist.append([span])
                    filtered_cat_list.append(cat_list[i])
                else:
                    log.warning(f"Invalid span for category '{cat}': {span}, skipping")
            else:
                log.warning(f"No span for category '{cat}', skipping")
        
        if not tokenspanlist or not filtered_cat_list:
            raise ValueError("Empty tokenspanlist or filtered_cat_list after filtering invalid spans")
        
        try:
            positive_map = create_positive_map_from_span(tokenized, tokenspanlist)
        except Exception as e:
            log.error(f"Error creating positive map: {e}")
            log.error(f"Tokenspanlist: {tokenspanlist}")
            raise
        
        # Pad positive_map to num_queries
        id_map = {i: i for i in range(len(filtered_cat_list))}
        new_pos_map = torch.zeros((self.num_queries, positive_map.shape[1]))
        for k, v in id_map.items():
            if k < len(positive_map):
                new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map.to(torch.float32)
        self.filtered_cat_list = filtered_cat_list + ["dummy_category"] * (self.num_queries - len(filtered_cat_list))
        # log.info(f"Positive map shape: {self.positive_map.shape}")
        # log.info(f"Filtered cat list length: {len(self.filtered_cat_list)}")

    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T
        prob = prob_to_label

        flat_prob = prob.view(out_logits.shape[0], -1)
        k = min(num_select, flat_prob.shape[1])
        topk_values, topk_indexes = torch.topk(flat_prob, k, dim=1)

        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        boxes = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]
        return results

class GroundingDino_Adaptation(torch.nn.Module):
    def __init__(self, detection_model, device="cuda"):
        super().__init__()
        self.detection_model = detection_model
        self.device = device
        self.caption = None
    
    def forward(self, images, h, w):
        batch = images.shape[0]
        captions = [self.caption for _ in range(batch)]
        with torch.no_grad():
            outputs = self.detection_model(images, captions=captions)
        return outputs

def process_batch_categories(args, detection_model, tokenizer, cat_list,
                           data_loader, coco_json, dataset):
    """
    Process categories by running detection for each image with its ground-truth class as the prompt.
    """
    all_detections = []
    category_map = {cat['id']: cat['name'] for cat in coco_json['categories']}

    torch.cuda.empty_cache()

    for images, targets, image_id in tqdm(data_loader, desc="Processing images"):
        if images is None or targets is None or image_id is None or not targets:
            continue
        
        image_id = int(image_id[0])
        target = targets[0]
        gt_category_ids = target['category_ids'].cpu().numpy()

        if len(gt_category_ids) == 0:
            continue

        # Get the ground-truth class name for this image
        gt_class_id = gt_category_ids[0]
        gt_class_name = category_map.get(gt_class_id)
        if not gt_class_name:
            continue

        # --- Create a simple, targeted prompt for the ground-truth class ---
        caption = preprocess_caption(f"{gt_class_name} .")
        detection_model.caption = caption

        # --- Instantiate a post-processor for this specific class ---
        try:
            postprocessor = PostProcessCIFAR100Grounding(
                num_select=10,  # We only need a few detections for one class
                cat_list=[gt_class_name], # Only provide the single ground-truth class
                tokenizer=tokenizer,
                num_queries=900
            )
        except ValueError as e:
            log.warning(f"Skipping '{gt_class_name}' due to postprocessor error: {e}")
            continue


        images = images.tensors.to(args.device)
        img_info = next(img for img in coco_json['images'] if img['id'] == image_id)

        with torch.no_grad():
            # --- Call the model without the problematic token_spans argument ---
            raw_outputs = detection_model(
                images,
                h=img_info['height'],
                w=img_info['width']
            )

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
        results = postprocessor(raw_outputs, orig_target_sizes)

        pred_detections = sv.Detections(
            xyxy=results[0]["boxes"].cpu().numpy(),
            # The class_id will be 0 since our cat_list has only one item
            class_id=results[0]["labels"].cpu().numpy(),
            confidence=results[0]["scores"].cpu().numpy()
        )

        # Filter by confidence and apply NMS
        mask = pred_detections.confidence > args.confidence_threshold
        pred_detections = pred_detections[mask]
        pred_detections = pred_detections.with_nms(threshold=args.nms_threshold)

        # Check if any of the top detections match the ground truth
        if len(pred_detections.xyxy) > 0:
            # Since the prompt was just the GT class, any detection is a "correct" class detection
            pred_box = pred_detections.xyxy[0] # Get the highest confidence box
            x1, y1, x2, y2 = pred_box
            xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            all_detections.append({
                "file_name": img_info["file_name"],
                "category": gt_class_name,
                "bbox": xywh
            })
            log.info(f"✓ CORRECT DETECTION: {gt_class_name} matched ground truth!")

    return all_detections


def main(args):
    cfg = SLConfig.fromfile(args.config_file)
    model, tokenizer = load_model(args.config_file, args.checkpoint_path, device=args.device)
    detection_model = GroundingDino_Adaptation(model, device=args.device)

    # Save CIFAR-100 images and create COCO-style JSON
    coco_json = save_cifar100_images(args.image_dir, args.num_images)
    dataset = CIFAR100Detection(args.image_dir, os.path.join(args.image_dir, "val.json"), transforms=data_transform)
    
    cat_list = [cat['name'] for cat in coco_json['categories']]
    log.info(f"Using {len(cat_list)} categories: {cat_list[:10]}...")
    
    # Create dataset and dataloader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=safe_collate_fn)

    output_data = {"case1": []}
    torch.cuda.set_device(args.device)

    # Simplified logic: directly call the corrected processing function
    all_detections = process_batch_categories(
        args, detection_model, tokenizer, cat_list,
        data_loader, coco_json, dataset
    )
    output_data["case1"].extend(all_detections)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    log.info(f"Saved correct detections to {args.output_path}")
    log.info(f"Total correct detections: {len(output_data['case1'])}")

if __name__ == "__main__":
    args = parse_args()
    main(args)