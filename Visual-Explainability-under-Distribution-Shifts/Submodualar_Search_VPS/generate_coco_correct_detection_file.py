# import os
# import json
# import numpy as np
# import torch
# import supervision as sv
# from PIL import Image
# from pycocotools.coco import COCO
# from tqdm import tqdm
# import argparse
# from torchvision.ops import box_convert
# import groundingdino.datasets.transforms as T
# from groundingdino.util import get_tokenlizer
# from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
# from groundingdino.util.slconfig import SLConfig
# from groundingdino.models import build_model
# from groundingdino.util.misc import clean_state_dict, collate_fn
# from torch.utils.data import DataLoader, Subset
# import torchvision
# import cv2
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes (80-class subset)
# COCO_CLASSES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
#     'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Correct COCO category IDs for the 80 classes
# COCO_ID_TO_NAME = {
#     1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
#     8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
#     14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
#     20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
#     27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
#     35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
#     40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
#     46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
#     52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
#     58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
#     64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
#     74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
#     80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
#     87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
# }

# # Map class indices to COCO category IDs
# CLASS_TO_COCO_ID = {COCO_CLASSES.index(name): coco_id for coco_id, name in COCO_ID_TO_NAME.items()}

# COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# data_transform = T.Compose([
#     T.RandomResize([800], max_size=1333),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate coco_groundingdino_correct_detections.json for COCO images")
#     parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config file")
#     parser.add_argument("--checkpoint_path", type=str, default="ckpt/groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint file")
#     parser.add_argument("--image_dir", type=str, default="datasets/val2017", help="COCO val2017 image directory")
#     parser.add_argument("--anno_path", type=str, default="datasets/annotations_trainval2017/annotations/instances_val2017.json", help="COCO val2017 annotations file")
#     parser.add_argument("--output_path", type=str, default="datasets/coco_groundingdino_correct_detections_again.json", help="Output JSON file path")
#     parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
#     parser.add_argument("--iou_threshold", type=float, default=0.6, help="IoU threshold for correct detections")
#     parser.add_argument("--confidence_threshold", type=float, default=0.9, help="Confidence threshold for detections")
#     parser.add_argument("--device", type=str, default="cuda:2", help="Device to run model (cuda or cpu)")
#     parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if not result.endswith("."):
#         result += "."
#     return result

# def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
#     args = SLConfig.fromfile(config_path)
#     args.device = device
#     model = build_model(args)
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
#     model.eval()
#     tokenizer = get_tokenlizer.get_tokenlizer(args.text_encoder_type if hasattr(args, 'text_encoder_type') else "bert-base-uncased")
#     return model.to(device), tokenizer

# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms):
#         super().__init__(img_folder, ann_file)
#         self._transforms = transforms

#     def __getitem__(self, idx):
#         img, target = super().__getitem__(idx)
#         w, h = img.size
#         boxes = [obj["bbox"] for obj in target]
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
#         boxes[:, 0::2].clamp_(min=0, max=w)
#         boxes[:, 1::2].clamp_(min=0, max=h)
#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
#         boxes = boxes[keep]

#         target_new = {}
#         image_id = self.ids[idx]
#         target_new["image_id"] = image_id
#         target_new["boxes"] = boxes
#         target_new["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target_new["category_ids"] = torch.as_tensor([obj["category_id"] for i, obj in enumerate(target) if keep[i]])

#         if self._transforms is not None:
#             img, target = self._transforms(img, target_new)

#         return img, target, image_id

# class PostProcessCocoGrounding(torch.nn.Module):
#     def __init__(self, num_select=50, coco_api=None, tokenizer=None):
#         super().__init__()
#         self.num_select = num_select
#         assert coco_api is not None
#         category_dict = coco_api.dataset['categories']
#         cat_list = [item['name'] for item in category_dict]
#         captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
#         print("Captions:", captions)
#         print("Token spans:", cat2tokenspan)
#         tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
#         positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist)

#         # Updated token indices based on tokenizer output
#         coco_classes_grounding_idx = {
#             'person': [0], 'bicycle': [9], 'car': [19], 'motorcycle': [25], 'airplane': [38],
#             'bus': [49], 'train': [55], 'truck': [63], 'boat': [71], 'traffic light': [78],
#             'fire hydrant': [94], 'stop sign': [109], 'parking meter': [121], 'bench': [137],
#             'bird': [145], 'cat': [152], 'dog': [158], 'horse': [164], 'sheep': [172],
#             'cow': [180], 'elephant': [186], 'bear': [197], 'zebra': [204], 'giraffe': [212],
#             'backpack': [222], 'umbrella': [233], 'handbag': [244], 'tie': [254], 'suitcase': [260],
#             'frisbee': [271], 'skis': [281], 'snowboard': [288], 'sports ball': [300], 'kite': [314],
#             'baseball bat': [321], 'baseball glove': [336], 'skateboard': [353], 'surfboard': [366],
#             'tennis racket': [378], 'bottle': [394], 'wine glass': [403], 'cup': [416], 'fork': [422],
#             'knife': [429], 'spoon': [437], 'bowl': [445], 'banana': [452], 'apple': [461],
#             'sandwich': [469], 'orange': [480], 'broccoli': [489], 'carrot': [500], 'hot dog': [509],
#             'pizza': [519], 'donut': [527], 'cake': [535], 'chair': [542], 'couch': [550],
#             'potted plant': [558], 'bed': [573], 'dining table': [579], 'toilet': [594], 'tv': [603],
#             'laptop': [608], 'mouse': [617], 'remote': [625], 'keyboard': [634], 'cell phone': [645],
#             'microwave': [658], 'oven': [670], 'toaster': [677], 'sink': [687], 'refrigerator': [694],
#             'book': [709], 'clock': [716], 'vase': [724], 'scissors': [731], 'teddy bear': [742],
#             'hair drier': [755], 'toothbrush': [768]
#         }

#         id_map = {}
#         for class_name, indices in coco_classes_grounding_idx.items():
#             if class_name in COCO_CLASSES:
#                 class_idx = COCO_CLASSES.index(class_name)
#                 for idx in indices:
#                     id_map[idx] = class_idx

#         max_token_idx = max(id_map.keys())
#         new_pos_map = torch.zeros((max_token_idx + 1, 256))  # Size based on max token index
#         for k, v in id_map.items():
#             new_pos_map[k] = positive_map[v]
#         self.positive_map = new_pos_map
#         self.id_map = id_map

#     def forward(self, outputs, target_sizes, not_to_xyxy=False):
#         num_select = self.num_select
#         out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
#         prob_to_token = out_logits.sigmoid()
#         pos_maps = self.positive_map.to(prob_to_token.device)
#         prob_to_label = prob_to_token @ pos_maps.T

#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2

#         prob = prob_to_label
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
#         scores = topk_values
#         topk_boxes = topk_indexes // prob.shape[2]
#         labels = topk_indexes % prob.shape[2]  # Token indices
#         print("Raw predicted token indices:", labels.cpu().numpy())

#         # Map token indices to class indices
#         class_labels = torch.zeros_like(labels)
#         for i, batch_labels in enumerate(labels):
#             for j, token_idx in enumerate(batch_labels):
#                 class_idx = self.id_map.get(token_idx.item(), -1)
#                 class_labels[i, j] = class_idx if class_idx >= 0 else 0  # Default to 'person' if not found

#         # Convert class indices to COCO category IDs
#         pred_class_names = [COCO_CLASSES[l] if 0 <= l < len(COCO_CLASSES) else "Unknown" for l in class_labels[0].cpu().numpy()]
#         coco_keys = list(COCO_ID_TO_NAME.keys())
#         pred_category_ids = [coco_keys[l] if 0 <= l < len(coco_keys) else 0 for l in class_labels[0].cpu().numpy()]
#         print("Predicted detections:", [(name, id) for name, id in zip(pred_class_names, pred_category_ids)])

#         if not_to_xyxy:
#             boxes = out_bbox
#         else:
#             boxes = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")

#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
#         img_h, img_w = target_sizes.unbind(1)
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]

#         results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, class_labels, boxes)]
#         return results

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda"):
#         super().__init__()
#         self.detection_model = detection_model
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         return outputs

# def main(args):
#     cfg = SLConfig.fromfile(args.config_file)
#     model, tokenizer = load_model(args.config_file, args.checkpoint_path, device=args.device)
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
#     caption = preprocess_caption(COCO_TEXT_PROMPT)
#     detection_model.caption = caption

#     dataset = CocoDetection(args.image_dir, args.anno_path, transforms=data_transform)
    
#     # Get list of image files in datasets/val2017
#     image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
#     if not image_files:
#         print(f"Error: No .jpg files found in {args.image_dir}")
#         return
    
#     # Extract COCO image IDs from filenames (e.g., "000000000285.jpg" -> 285)
#     image_ids = []
#     for img_file in image_files:
#         filename = os.path.basename(img_file)
#         try:
#             img_id = int(filename.split('.')[0])
#             if img_id in dataset.ids:  # Ensure ID exists in COCO annotations
#                 image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid filename format for {filename}, skipping")
#             continue
    
#     if not image_ids:
#         print(f"Error: No valid image IDs found in {args.image_dir} that match COCO annotations")
#         return
    
#     # Randomly select up to num_images (500) IDs
#     num_images = min(args.num_images, len(image_ids))
#     try:
#         selected_ids = np.random.choice(image_ids, size=num_images, replace=False).tolist()
#         indices = [dataset.ids.index(img_id) for img_id in selected_ids]
#     except ValueError as e:
#         print(f"Error: One or more selected image IDs not found in dataset: {e}")
#         return
    
#     print(f"Selected {len(indices)} images from {args.image_dir}")
#     subset_dataset = Subset(dataset, indices)
#     data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

#     postprocessor = PostProcessCocoGrounding(num_select=10, coco_api=dataset.coco, tokenizer=tokenizer)

#     coco = COCO(args.anno_path)
#     output_data = {"case1": []}
    
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     for images, targets, image_id in tqdm(data_loader, total=len(data_loader), desc="Processing images"):
#         image_id = int(image_id[0])
#         img_info = coco.loadImgs([image_id])[0]
#         img_file = os.path.join(args.image_dir, img_info["file_name"])
        
#         image = cv2.imread(img_file)
#         if image is None:
#             print(f"Failed to load image: {img_file}")
#             continue

#         images = images.tensors.to(args.device)
#         with torch.no_grad():
#             raw_outputs = detection_model.detection_model(images, captions=[detection_model.caption])
#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(args.device)
#         results = postprocessor(raw_outputs, orig_target_sizes)

#         pred_detections = sv.Detections(
#             xyxy=results[0]["boxes"].cpu().numpy(),
#             class_id=results[0]["labels"].cpu().numpy(),
#             confidence=results[0]["scores"].cpu().numpy()
#         )
#         mask = pred_detections.confidence > args.confidence_threshold
#         pred_detections = sv.Detections(
#             xyxy=pred_detections.xyxy[mask],
#             class_id=pred_detections.class_id[mask],
#             confidence=pred_detections.confidence[mask]
#         )
#         pred_detections = pred_detections.with_nms(threshold=0.2, class_agnostic=False)

#         gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
#         gt_boxes = np.array([ann["bbox"] for ann in gt_anns])
#         if len(gt_boxes) == 0:
#             print(f"No annotations for image ID {image_id}")
#             continue
#         gt_boxes_xyxy = gt_boxes.copy()
#         gt_boxes_xyxy[:, 2:] += gt_boxes_xyxy[:, :2]
#         gt_class_ids = np.array([ann["category_id"] for ann in gt_anns])
#         gt_class_names = [COCO_ID_TO_NAME.get(id, "Unknown") for id in gt_class_ids]
#         print(f"Ground truth detections for image {image_id}:", [(name, int(id)) for name, id in zip(gt_class_names, gt_class_ids)])

#         gt_detections = sv.Detections(
#             xyxy=gt_boxes_xyxy,
#             class_id=gt_class_ids
#         )

#         iou_matrix = sv.detection.utils.box_iou_batch(pred_detections.xyxy, gt_detections.xyxy)
#         for pred_idx, (pred_box, pred_class, pred_conf) in enumerate(zip(pred_detections.xyxy, pred_detections.class_id, pred_detections.confidence)):
#             # Map class index to COCO category ID
#             coco_id = CLASS_TO_COCO_ID.get(pred_class, 0)
#             if coco_id not in COCO_ID_TO_NAME:
#                 print(f"Skipping invalid class index {pred_class} (COCO ID {coco_id}) for image {image_id}")
#                 continue
#             max_iou_idx = np.argmax(iou_matrix[pred_idx]) if iou_matrix[pred_idx].size > 0 else None
#             iou = iou_matrix[pred_idx, max_iou_idx] if max_iou_idx is not None else 0.0
#             print(f"Prediction for image {image_id}: {COCO_ID_TO_NAME[coco_id]} (ID: {coco_id}), Confidence: {pred_conf:.2f}, IoU: {iou:.2f}")
#             if max_iou_idx is not None and iou >= args.iou_threshold:
#                 gt_class = gt_detections.class_id[max_iou_idx]
#                 if coco_id == gt_class:
#                     gt_box = gt_boxes[max_iou_idx]
#                     output_data["case1"].append({
#                         "file_name": img_info["file_name"],
#                         "category": COCO_ID_TO_NAME[coco_id],
#                         "bbox": [float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3])]
#                     })

#         for i, (box, class_id, conf) in enumerate(zip(pred_detections.xyxy, pred_detections.class_id, pred_detections.confidence)):
#             coco_id = CLASS_TO_COCO_ID.get(class_id, 0)
#             x1, y1, x2, y2 = box.astype(int)
#             label = f"{COCO_ID_TO_NAME.get(coco_id, 'Unknown')}: {conf:.2f}"
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         cv2.imwrite(f"output_{image_id}.jpg", image)
#         print(f"Saved visualization to output_{image_id}.jpg")

#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
#     with open(args.output_path, "w", encoding="utf-8") as f:
#         json.dump(output_data, f, indent=4, ensure_ascii=False)
#     print(f"Saved correct detections to {args.output_path}")
#     print(f"Total correct detections: {len(output_data['case1'])}")

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

import os
import json
import numpy as np
import torch
import supervision as sv
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T
from groundingdino.util import get_tokenlizer
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict, collate_fn
from torch.utils.data import DataLoader, Subset
import torchvision
import cv2
import glob

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define COCO classes (80-class subset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Correct COCO category IDs for the 80 classes
COCO_ID_TO_NAME = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
    8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
    14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Map class indices to COCO category IDs
CLASS_TO_COCO_ID = {COCO_CLASSES.index(name): coco_id for coco_id, name in COCO_ID_TO_NAME.items()}

COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

data_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def parse_args():
    parser = argparse.ArgumentParser(description="Generate coco_groundingdino_correct_detections.json for COCO images with single detections only")
    parser.add_argument("--config_file", type=str, default="config/GroundingDINO_SwinT_OGC.py", help="Path to Grounding DINO config file")
    parser.add_argument("--checkpoint_path", type=str, default="ckpt/groundingdino_swint_ogc.pth", help="Path to Grounding DINO checkpoint file")
    parser.add_argument("--image_dir", type=str, default="datasets/val2017", help="COCO val2017 image directory")
    parser.add_argument("--anno_path", type=str, default="datasets/annotations_trainval2017/annotations/instances_val2017.json", help="COCO val2017 annotations file")
    parser.add_argument("--output_path", type=str, default="datasets/coco_groundingdino_single_detections.json", help="Output JSON file path")
    parser.add_argument("--num_images", type=int, default=500, help="Number of images to process")
    parser.add_argument("--iou_threshold", type=float, default=0.6, help="IoU threshold for correct detections")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to run model (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    return parser.parse_args()

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result

def load_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    tokenizer = get_tokenlizer.get_tokenlizer(args.text_encoder_type if hasattr(args, 'text_encoder_type') else "bert-base-uncased")
    return model.to(device), tokenizer

def filter_single_detection_images(coco, image_ids):
    """Filter images to only include those with exactly one annotation/bounding box"""
    single_detection_ids = []
    skipped_count = 0
    
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        # Filter valid annotations (non-zero area bounding boxes)
        valid_anns = []
        for ann in anns:
            bbox = ann["bbox"]
            if bbox[2] > 0 and bbox[3] > 0:  # width > 0 and height > 0
                valid_anns.append(ann)
        
        # Only keep images with exactly one valid annotation
        if len(valid_anns) == 1:
            single_detection_ids.append(img_id)
        else:
            skipped_count += 1
    
    print(f"Filtered to {len(single_detection_ids)} images with single detections")
    print(f"Skipped {skipped_count} images with multiple or zero detections")
    
    return single_detection_ids

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_new["category_ids"] = torch.as_tensor([obj["category_id"] for i, obj in enumerate(target) if keep[i]])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target, image_id

class PostProcessCocoGrounding(torch.nn.Module):
    def __init__(self, num_select=50, coco_api=None, tokenizer=None):
        super().__init__()
        self.num_select = num_select
        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        print("Captions:", captions)
        print("Token spans:", cat2tokenspan)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist)

        # Updated token indices based on tokenizer output
        coco_classes_grounding_idx = {
            'person': [0], 'bicycle': [9], 'car': [19], 'motorcycle': [25], 'airplane': [38],
            'bus': [49], 'train': [55], 'truck': [63], 'boat': [71], 'traffic light': [78],
            'fire hydrant': [94], 'stop sign': [109], 'parking meter': [121], 'bench': [137],
            'bird': [145], 'cat': [152], 'dog': [158], 'horse': [164], 'sheep': [172],
            'cow': [180], 'elephant': [186], 'bear': [197], 'zebra': [204], 'giraffe': [212],
            'backpack': [222], 'umbrella': [233], 'handbag': [244], 'tie': [254], 'suitcase': [260],
            'frisbee': [271], 'skis': [281], 'snowboard': [288], 'sports ball': [300], 'kite': [314],
            'baseball bat': [321], 'baseball glove': [336], 'skateboard': [353], 'surfboard': [366],
            'tennis racket': [378], 'bottle': [394], 'wine glass': [403], 'cup': [416], 'fork': [422],
            'knife': [429], 'spoon': [437], 'bowl': [445], 'banana': [452], 'apple': [461],
            'sandwich': [469], 'orange': [480], 'broccoli': [489], 'carrot': [500], 'hot dog': [509],
            'pizza': [519], 'donut': [527], 'cake': [535], 'chair': [542], 'couch': [550],
            'potted plant': [558], 'bed': [573], 'dining table': [579], 'toilet': [594], 'tv': [603],
            'laptop': [608], 'mouse': [617], 'remote': [625], 'keyboard': [634], 'cell phone': [645],
            'microwave': [658], 'oven': [670], 'toaster': [677], 'sink': [687], 'refrigerator': [694],
            'book': [709], 'clock': [716], 'vase': [724], 'scissors': [731], 'teddy bear': [742],
            'hair drier': [755], 'toothbrush': [768]
        }

        id_map = {}
        for class_name, indices in coco_classes_grounding_idx.items():
            if class_name in COCO_CLASSES:
                class_idx = COCO_CLASSES.index(class_name)
                for idx in indices:
                    id_map[idx] = class_idx

        max_token_idx = max(id_map.keys())
        new_pos_map = torch.zeros((max_token_idx + 1, 256))  # Size based on max token index
        for k, v in id_map.items():
            new_pos_map[k] = positive_map[v]
        self.positive_map = new_pos_map
        self.id_map = id_map

    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]  # Token indices
        print("Raw predicted token indices:", labels.cpu().numpy())

        # Map token indices to class indices
        class_labels = torch.zeros_like(labels)
        for i, batch_labels in enumerate(labels):
            for j, token_idx in enumerate(batch_labels):
                class_idx = self.id_map.get(token_idx.item(), -1)
                class_labels[i, j] = class_idx if class_idx >= 0 else 0  # Default to 'person' if not found

        # Convert class indices to COCO category IDs
        pred_class_names = [COCO_CLASSES[l] if 0 <= l < len(COCO_CLASSES) else "Unknown" for l in class_labels[0].cpu().numpy()]
        coco_keys = list(COCO_ID_TO_NAME.keys())
        pred_category_ids = [coco_keys[l] if 0 <= l < len(coco_keys) else 0 for l in class_labels[0].cpu().numpy()]
        print("Predicted detections:", [(name, id) for name, id in zip(pred_class_names, pred_category_ids)])

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, class_labels, boxes)]
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

def main(args):
    cfg = SLConfig.fromfile(args.config_file)
    model, tokenizer = load_model(args.config_file, args.checkpoint_path, device=args.device)
    detection_model = GroundingDino_Adaptation(model, device=args.device)
    caption = preprocess_caption(COCO_TEXT_PROMPT)
    detection_model.caption = caption

    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=data_transform)
    coco = COCO(args.anno_path)
    
    # Get list of image files in datasets/val2017
    image_files = glob.glob(os.path.join(args.image_dir, "*.jpg"))
    if not image_files:
        print(f"Error: No .jpg files found in {args.image_dir}")
        return
    
    # Extract COCO image IDs from filenames (e.g., "000000000285.jpg" -> 285)
    image_ids = []
    for img_file in image_files:
        filename = os.path.basename(img_file)
        try:
            img_id = int(filename.split('.')[0])
            if img_id in dataset.ids:  # Ensure ID exists in COCO annotations
                image_ids.append(img_id)
        except ValueError:
            print(f"Warning: Invalid filename format for {filename}, skipping")
            continue
    
    if not image_ids:
        print(f"Error: No valid image IDs found in {args.image_dir} that match COCO annotations")
        return
    
    # Filter to only images with single detections
    single_detection_ids = filter_single_detection_images(coco, image_ids)
    
    if not single_detection_ids:
        print("Error: No images found with single detections")
        return
    
    # Randomly select up to num_images from single detection images
    num_images = min(args.num_images, len(single_detection_ids))
    try:
        selected_ids = np.random.choice(single_detection_ids, size=num_images, replace=False).tolist()
        indices = [dataset.ids.index(img_id) for img_id in selected_ids]
    except ValueError as e:
        print(f"Error: One or more selected image IDs not found in dataset: {e}")
        return
    
    print(f"Selected {len(indices)} single-detection images from {args.image_dir}")
    subset_dataset = Subset(dataset, indices)
    data_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    postprocessor = PostProcessCocoGrounding(num_select=10, coco_api=dataset.coco, tokenizer=tokenizer)

    output_data = {"case1": []}
    
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    
    for images, targets, image_id in tqdm(data_loader, total=len(data_loader), desc="Processing single-detection images"):
        image_id = int(image_id[0])
        img_info = coco.loadImgs([image_id])[0]
        img_file = os.path.join(args.image_dir, img_info["file_name"])
        
        image = cv2.imread(img_file)
        if image is None:
            print(f"Failed to load image: {img_file}")
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
        pred_detections = pred_detections.with_nms(threshold=0.2, class_agnostic=False)

        gt_anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        gt_boxes = np.array([ann["bbox"] for ann in gt_anns])
        
        # Since we filtered for single detections, we should have exactly one annotation
        if len(gt_boxes) != 1:
            print(f"Warning: Expected 1 annotation but found {len(gt_boxes)} for image ID {image_id}")
            continue
            
        gt_boxes_xyxy = gt_boxes.copy()
        gt_boxes_xyxy[:, 2:] += gt_boxes_xyxy[:, :2]
        gt_class_ids = np.array([ann["category_id"] for ann in gt_anns])
        gt_class_names = [COCO_ID_TO_NAME.get(id, "Unknown") for id in gt_class_ids]
        print(f"Ground truth detection for image {image_id}: {gt_class_names[0]} (ID: {int(gt_class_ids[0])})")

        gt_detections = sv.Detections(
            xyxy=gt_boxes_xyxy,
            class_id=gt_class_ids
        )

        iou_matrix = sv.detection.utils.box_iou_batch(pred_detections.xyxy, gt_detections.xyxy)
        for pred_idx, (pred_box, pred_class, pred_conf) in enumerate(zip(pred_detections.xyxy, pred_detections.class_id, pred_detections.confidence)):
            # Map class index to COCO category ID
            coco_id = CLASS_TO_COCO_ID.get(pred_class, 0)
            if coco_id not in COCO_ID_TO_NAME:
                print(f"Skipping invalid class index {pred_class} (COCO ID {coco_id}) for image {image_id}")
                continue
            max_iou_idx = np.argmax(iou_matrix[pred_idx]) if iou_matrix[pred_idx].size > 0 else None
            iou = iou_matrix[pred_idx, max_iou_idx] if max_iou_idx is not None else 0.0
            print(f"Prediction for image {image_id}: {COCO_ID_TO_NAME[coco_id]} (ID: {coco_id}), Confidence: {pred_conf:.2f}, IoU: {iou:.2f}")
            if max_iou_idx is not None and iou >= args.iou_threshold:
                gt_class = gt_detections.class_id[max_iou_idx]
                if coco_id == gt_class:
                    gt_box = gt_boxes[max_iou_idx]
                    output_data["case1"].append({
                        "file_name": img_info["file_name"],
                        "category": COCO_ID_TO_NAME[coco_id],
                        "bbox": [float(gt_box[0]), float(gt_box[1]), float(gt_box[2]), float(gt_box[3])]
                    })

        # Visualization with single detection emphasis
        for i, (box, class_id, conf) in enumerate(zip(pred_detections.xyxy, pred_detections.class_id, pred_detections.confidence)):
            coco_id = CLASS_TO_COCO_ID.get(class_id, 0)
            x1, y1, x2, y2 = box.astype(int)
            label = f"{COCO_ID_TO_NAME.get(coco_id, 'Unknown')}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ground truth box for comparison
        gt_box = gt_boxes_xyxy[0].astype(int)
        cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)
        cv2.putText(image, f"GT: {gt_class_names[0]}", (gt_box[0], gt_box[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(f"output_single_{image_id}.jpg", image)
        print(f"Saved visualization to output_single_{image_id}.jpg")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Saved correct single detections to {args.output_path}")
    print(f"Total correct single detections: {len(output_data['case1'])}")

if __name__ == "__main__":
    args = parse_args()
    main(args)