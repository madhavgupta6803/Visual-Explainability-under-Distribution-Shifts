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
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.ops import box_convert
import logging
import re
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data transform
data_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def parse_args():
    parser = argparse.ArgumentParser(description="Generate INaturalist_correct_Detection_grounding.json with predicted bounding boxes")
    parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config file")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint file")
    parser.add_argument("--image_dir", type=str, default="datasets/INaturalist", help="iNaturalist val2017 image directory")
    parser.add_argument("--anno_path", type=str, default="datasets/val.json", help="iNaturalist val2017 annotations file")
    parser.add_argument("--output_path", type=str, default="datasets/INaturalist_correct_Detection_grounding.json", help="Output JSON file path")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
    parser.add_argument("--confidence_threshold", type=float, default=0.1, help="Confidence threshold for predictions")
    parser.add_argument("--nms_threshold", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run model (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--max_categories", type=int, default=66, help="Maximum number of categories for text prompt (reduced default)")
    parser.add_argument("--batch_processing", action="store_true", help="Enable batch processing to handle large category lists")
    parser.add_argument("--categories_per_batch", type=int, default=3, help="Number of categories per batch when batch processing is enabled")
    parser.add_argument("--folder_range_start", type=str, default="00000", help="Start folder range")
    parser.add_argument("--folder_range_end", type=str, default="00065", help="End folder range (reduced default)")
    return parser.parse_args()

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result

def safe_collate_fn(batch):
    """Custom collate function that handles None values"""
    # Filter out None entries
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None, None
    
    # Use original collate_fn for valid entries
    from groundingdino.util.misc import collate_fn
    return collate_fn(batch)

def filter_images_by_folder_range(coco_json, img_folder, start_folder="00000", end_folder="00100", max_images=500):
    """Filter images that belong to folders within the specified range and collect their categories"""
    valid_images = []
    valid_category_ids = set()
    
    # Get all available directories in the specified range
    if os.path.exists(img_folder):
        all_dirs = [d for d in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, d))]
        # Filter directories by range - extract the numeric part for comparison
        valid_dirs = []
        for d in all_dirs:
            # Extract the first 5 characters as the folder number
            folder_num = d[:5]
            try:
                if start_folder <= folder_num <= end_folder:
                    valid_dirs.append(d)
            except:
                # Skip directories that don't follow the expected naming pattern
                continue
        
        valid_dirs.sort()  # Sort for consistent ordering
        log.info(f"Found {len(valid_dirs)} directories in range {start_folder}-{end_folder}")
        log.info(f"Sample directories: {valid_dirs[:5]}")
        
        # Log the actual range of directories found
        if valid_dirs:
            log.info(f"Directory range: {valid_dirs[0][:5]} to {valid_dirs[-1][:5]}")
    else:
        log.error(f"Image folder {img_folder} does not exist")
        return [], set()
    
    if not valid_dirs:
        log.error(f"No directories found in range {start_folder}-{end_folder}")
        return [], set()
    
    # Find images that belong to these directories and collect their categories
    images_processed = 0
    for img in coco_json['images']:
        if images_processed >= max_images:
            break
            
        img_found = False
        for valid_dir in valid_dirs:
            # Check if image path contains this directory or if we can find the image in this directory
            if valid_dir in img['file_name'] or os.path.exists(os.path.join(img_folder, valid_dir, os.path.basename(img['file_name']))):
                valid_images.append(img)
                
                # Collect the category IDs for this specific image
                img_anns = [ann for ann in coco_json['annotations'] if ann['image_id'] == img['id']]
                for ann in img_anns:
                    valid_category_ids.add(ann['category_id'])
                
                img_found = True
                images_processed += 1
                break
        
        if not img_found:
            # Try a more thorough search for this image in valid directories
            base_filename = os.path.basename(img['file_name'])
            for valid_dir in valid_dirs:
                dir_path = os.path.join(img_folder, valid_dir)
                if os.path.isdir(dir_path):
                    potential_path = os.path.join(dir_path, base_filename)
                    if os.path.exists(potential_path):
                        valid_images.append(img)
                        # Collect categories for this image
                        img_anns = [ann for ann in coco_json['annotations'] if ann['image_id'] == img['id']]
                        for ann in img_anns:
                            valid_category_ids.add(ann['category_id'])
                        images_processed += 1
                        break
            
            if images_processed >= max_images:
                break
    
    log.info(f"Found {len(valid_images)} valid images from folder range {start_folder}-{end_folder}")
    log.info(f"These images contain {len(valid_category_ids)} unique categories")
    
    # Log some statistics about the categories found
    if valid_category_ids:
        log.info(f"Category ID range in selected folders: {min(valid_category_ids)} to {max(valid_category_ids)}")
        
        # Find category names for logging
        category_names = []
        for cat in coco_json['categories']:
            if cat['id'] in valid_category_ids:
                category_names.append(cat['name'])
        log.info(f"Sample category names from selected folders: {category_names[:10]}")
    
    return valid_images[:max_images], valid_category_ids

def normalize_class_name(class_name: str) -> str:
    """Normalize class names, ensuring non-empty output."""
    norm_name = re.sub(r'[^a-zA-Z0-9\s]', '', class_name.lower().strip())
    return norm_name if norm_name else class_name.lower().replace(' ', '_')

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
            # Current batch is full, start a new one
            batches.append(current_batch)
            current_batch = [cat]
        else:
            current_batch.append(cat)
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches

def find_image_path(img_folder, img_info, cat_dir_map=None, image_id=None, coco_data=None):
    """Helper function to find the correct image path for iNaturalist dataset"""
    base_filename = os.path.basename(img_info['file_name'])
    
    # Strategy 1: Try original path from JSON
    img_path = os.path.join(img_folder, img_info['file_name'])
    if os.path.exists(img_path):
        return img_path
    
    # Strategy 2: Try without 'val/' prefix
    img_path_alt = os.path.join(img_folder, img_info['file_name'].replace('val/', '', 1))
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
    
    # Strategy 4: Try category-based path using cat_dir_map if available
    if cat_dir_map and image_id and coco_data:
        anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        if anns and anns[0]['category_id'] in cat_dir_map:
            img_path_cat = os.path.join(img_folder, cat_dir_map[anns[0]['category_id']], base_filename)
            if os.path.exists(img_path_cat):
                return img_path_cat
    
    # Strategy 5: Try just the basename in the main folder
    img_path_base = os.path.join(img_folder, base_filename)
    if os.path.exists(img_path_base):
        return img_path_base
    
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

class INaturalistDetection(torchvision.datasets.VisionDataset):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder)
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.img_folder = img_folder
        self.transforms = transforms
        self.ids = [img['id'] for img in self.coco['images']]
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}
        self.cat_list = [cat['name'] for cat in self.coco['categories']]
        self.cat_dir_map = {cat['id']: cat['image_dir_name'] for cat in self.coco['categories']}
        log.info(f"Loaded {len(self.cat_list)} categories, e.g., {self.cat_list[:5]}")

        # Test image path resolution
        self._test_image_paths()

    def _test_image_paths(self):
        """Test and log image path resolution strategies"""
        sample_img = self.coco['images'][0]
        log.info(f"Testing image path resolution for: {sample_img['file_name']}")
        
        found_path = find_image_path(self.img_folder, sample_img, self.cat_dir_map, 
                                   sample_img['id'], self.coco)
        
        if found_path:
            log.info(f"Successfully resolved image path: {found_path}")
        else:
            log.error(f"Could not resolve path for sample image: {sample_img['file_name']}")
            # List available directories for debugging
            if os.path.exists(self.img_folder):
                dirs = [d for d in os.listdir(self.img_folder) if os.path.isdir(os.path.join(self.img_folder, d))][:10]
                log.info(f"Available directories in {self.img_folder}: {dirs}")

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = next(img for img in self.coco['images'] if img['id'] == img_id)
        
        # Use helper function to find image path
        img_path = find_image_path(self.img_folder, img_info, self.cat_dir_map, img_id, self.coco)
        
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

class PostProcessINaturalistGrounding(torch.nn.Module):
    def __init__(self, num_select=50, cat_list=None, tokenizer=None):
        super().__init__()
        self.num_select = num_select
        assert cat_list is not None
        
        # Normalize category names
        normalized_cat_list = []
        for cat in cat_list:
            norm_name = normalize_class_name(cat)
            if norm_name:
                normalized_cat_list.append(norm_name)
            else:
                log.warning(f"Skipping empty normalized name for '{cat}'")
        
        # Use the original GroundingDINO function to build captions and token spans
        captions, cat2tokenspan = build_captions_and_token_span(normalized_cat_list, True)
        
        log.info(f"Captions: {captions[:100]}...")
        log.info(f"Token spans sample: {list(cat2tokenspan.items())[:5]}")
        log.info(f"Valid categories: {len(normalized_cat_list)}, Sample: {normalized_cat_list[:5]}")
        
        # Tokenize the captions
        tokenized = tokenizer(captions)
        token_count = len(tokenized['input_ids'])
        log.info(f"Token count: {token_count}")
        
        if token_count > 512:
            raise ValueError(f"Token count {token_count} exceeds BERT limit of 512. Current categories: {len(cat_list)}. Try reducing --max_categories or --categories_per_batch.")
        
        # Create token span list in the correct format
        tokenspanlist = []
        filtered_cat_list = []
        
        for i, cat in enumerate(normalized_cat_list):
            if cat in cat2tokenspan:
                span = cat2tokenspan[cat]
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    # Convert to list of token indices if it's a range
                    if len(span) == 2 and isinstance(span[0], int) and isinstance(span[1], int):
                        token_indices = list(range(span[0], span[1]))
                        tokenspanlist.append(token_indices)
                        filtered_cat_list.append(cat_list[i])  # Keep original category name
                    else:
                        tokenspanlist.append(span)
                        filtered_cat_list.append(cat_list[i])
                else:
                    log.warning(f"Invalid span for category '{cat}': {span}, skipping")
        
        if not tokenspanlist or not filtered_cat_list:
            raise ValueError("Empty tokenspanlist or filtered_cat_list after filtering invalid spans")
        
        # Create positive map
        positive_map = create_positive_map_from_span(tokenized, tokenspanlist)
        
        # Create identity mapping for categories
        id_map = {i: i for i in range(len(filtered_cat_list))}
        new_pos_map = torch.zeros((len(filtered_cat_list), positive_map.shape[1]))
        
        for k, v in id_map.items():
            if k < len(positive_map):
                new_pos_map[v] = positive_map[k]
        
        self.positive_map = new_pos_map
        self.filtered_cat_list = filtered_cat_list

    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        boxes = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
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

def process_batch_categories(args, detection_model, tokenizer, cat_batch, cat_id_to_filtered_idx_batch, 
                           data_loader, coco_json, dataset, valid_categories):
    """Process a batch of categories"""
    log.info(f"Processing batch with {len(cat_batch)} categories")
    
    # Create text prompt using normalized names
    normalized_names = [normalize_class_name(cat) for cat in cat_batch]
    INAT_TEXT_PROMPT = " . ".join(normalized_names) + " ."
    caption = preprocess_caption(INAT_TEXT_PROMPT)
    detection_model.caption = caption
    
    # Estimate token count
    token_count = estimate_token_count(cat_batch, tokenizer)
    log.info(f"Text prompt length: {len(INAT_TEXT_PROMPT)} characters, estimated tokens: {token_count}")
    
    if token_count > 512:
        log.warning(f"Token count {token_count} still exceeds limit for this batch. Skipping.")
        return []
    
    postprocessor = PostProcessINaturalistGrounding(num_select=50, cat_list=cat_batch, tokenizer=tokenizer)
    
    batch_detections = []
    
    torch.cuda.empty_cache()  # Clear GPU memory before processing
    
    for images, targets, image_id in tqdm(data_loader, desc=f"Processing batch"):
        if images is None or targets is None or image_id is None:
            continue
            
        if len(image_id) == 0:
            continue
            
        image_id = int(image_id[0])
        img_info = next(img for img in coco_json['images'] if img['id'] == image_id)
        
        img_file = find_image_path(args.image_dir, img_info, dataset.cat_dir_map, image_id, coco_json)
        if img_file is None:
            continue
        
        image = cv2.imread(img_file)
        if image is None:
            continue

        images = images.tensors.to(args.device)
        with torch.no_grad():
            raw_outputs = detection_model.detection_model(images, captions=[detection_model.caption])
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
        results = postprocessor(raw_outputs, orig_target_sizes)

        pred_detections = sv.Detections(
            xyxy=results[0]["boxes"].cpu().numpy(),
            class_id=results[0]["labels"].cpu().numpy(),
            confidence=results[0]["scores"].cpu().numpy()
        )
        mask = pred_detections.confidence > args.confidence_threshold
        pred_detections = sv.Detections(
            xyxy=pred_detections.xyxy[mask],
            class_id=pred_detections.class_id[mask],
            confidence=pred_detections.confidence[mask]
        )
        pred_detections = pred_detections.with_nms(threshold=args.nms_threshold, class_agnostic=False)

        anns = [ann for ann in coco_json['annotations'] if ann['image_id'] == image_id]
        if not anns:
            continue
            
        gt_class_ids = np.array([ann["category_id"] for ann in anns])

        for pred_box, pred_class_idx, pred_conf in zip(pred_detections.xyxy, pred_detections.class_id, pred_detections.confidence):
            if pred_class_idx >= len(postprocessor.filtered_cat_list):
                continue
                
            pred_class_name = postprocessor.filtered_cat_list[pred_class_idx]
            
            # Find the original category ID
            pred_class_id = None
            for cat in valid_categories:
                if cat['name'] == pred_class_name and cat['id'] in cat_id_to_filtered_idx_batch:
                    if cat_id_to_filtered_idx_batch[cat['id']] == pred_class_idx:
                        pred_class_id = cat['id']
                        break
            
            if pred_class_id is None:
                continue
                
            if pred_class_id in gt_class_ids:
                x1, y1, x2, y2 = pred_box
                xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                batch_detections.append({
                    "file_name": img_info["file_name"],
                    "category": pred_class_name,
                    "bbox": xywh
                })
                log.info(f"✓ CORRECT DETECTION: {pred_class_name} matched ground truth!")
    
    return batch_detections

def main(args):
    cfg = SLConfig.fromfile(args.config_file)
    model, tokenizer = load_model(args.config_file, args.checkpoint_path, device=args.device)
    detection_model = GroundingDino_Adaptation(model, device=args.device)

    dataset = INaturalistDetection(args.image_dir, args.anno_path, transforms=data_transform)
    with open(args.anno_path, 'r') as f:
        coco_json = json.load(f)
    
    # STEP 1: Filter images to only include those from the specified folder range
    valid_images, valid_category_ids = filter_images_by_folder_range(
        coco_json, args.image_dir, args.folder_range_start, args.folder_range_end, args.num_images
    )
    
    if not valid_images:
        log.error("No valid images found in the specified folder range. Exiting.")
        return
    
    # STEP 2: Filter categories to ONLY include those that appear in our selected folder range
    # This ensures we only process categories that actually exist in the selected folders
    valid_categories = []
    for cat in coco_json['categories']:
        if cat['id'] in valid_category_ids:
            valid_categories.append(cat)
    
    # Sort by category ID to maintain consistent ordering
    valid_categories.sort(key=lambda x: x['id'])
    
    # STEP 3: Apply max_categories limit to the folder-filtered categories
    selected_categories = valid_categories[:args.max_categories]
    cat_list = [cat['name'] for cat in selected_categories]
    
    log.info(f"Folder range {args.folder_range_start}-{args.folder_range_end} contains {len(valid_category_ids)} unique categories")
    log.info(f"Using {len(cat_list)} categories from the selected folder range (limited by --max_categories)")
    log.info(f"Sample categories from selected folders: {cat_list[:10]}")
    
    # STEP 4: Verify that all selected categories actually exist in our folder range
    category_verification = {}
    for cat in selected_categories:
        category_verification[cat['id']] = cat['name']
    
    log.info(f"Category verification - all {len(category_verification)} categories are from folders {args.folder_range_start}-{args.folder_range_end}")
    
    # STEP 5: Check if we need batch processing
    estimated_tokens = estimate_token_count(cat_list, tokenizer)
    log.info(f"Estimated token count for all selected categories: {estimated_tokens}")
    
    if estimated_tokens > 512 and not args.batch_processing:
        log.error(f"Token count {estimated_tokens} exceeds BERT limit. Enable --batch_processing or reduce --max_categories to {int(len(cat_list) * 400 / estimated_tokens)}")
        return
    
    # STEP 6: Create dataset subset with only images from selected folder range
    valid_image_ids = [img['id'] for img in valid_images]
    
    # Find indices in the dataset that correspond to our valid images
    indices = []
    for img_id in valid_image_ids:
        try:
            idx = dataset.ids.index(img_id)
            indices.append(idx)
        except ValueError:
            continue
    
    log.info(f"Successfully found {len(indices)} images to process from folders {args.folder_range_start}-{args.folder_range_end}")
    subset_dataset = Subset(dataset, indices)
    data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=safe_collate_fn)

    output_data = {"case1": []}
    
    torch.cuda.set_device(args.device)
    
    if args.batch_processing and estimated_tokens > 512:
        # Split categories into batches
        category_batches = split_categories_by_token_limit(cat_list, tokenizer, max_tokens=400)
        log.info(f"Split {len(cat_list)} categories into {len(category_batches)} batches for processing")
        
        for batch_idx, cat_batch in enumerate(category_batches):
            log.info(f"Processing batch {batch_idx + 1}/{len(category_batches)} with {len(cat_batch)} categories")
            
            # Create mapping for this batch - only use categories from selected folders
            cat_id_to_filtered_idx_batch = {}
            batch_valid_categories = []
            for idx, cat_name in enumerate(cat_batch):
                for cat in selected_categories:  # Use selected_categories instead of valid_categories
                    if cat['name'] == cat_name:
                        cat_id_to_filtered_idx_batch[cat['id']] = idx
                        batch_valid_categories.append(cat)
                        break
            
            batch_detections = process_batch_categories(
                args, detection_model, tokenizer, cat_batch, cat_id_to_filtered_idx_batch,
                data_loader, coco_json, dataset, batch_valid_categories
            )
            
            output_data["case1"].extend(batch_detections)
            log.info(f"Batch {batch_idx + 1} contributed {len(batch_detections)} correct detections")
    
    else:
        # Process all categories at once (original method)
        cat_id_to_filtered_idx = {}
        for idx, cat in enumerate(selected_categories):  # Use selected_categories
            cat_id_to_filtered_idx[cat['id']] = idx
        
        all_detections = process_batch_categories(
            args, detection_model, tokenizer, cat_list, cat_id_to_filtered_idx,
            data_loader, coco_json, dataset, selected_categories  # Use selected_categories
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