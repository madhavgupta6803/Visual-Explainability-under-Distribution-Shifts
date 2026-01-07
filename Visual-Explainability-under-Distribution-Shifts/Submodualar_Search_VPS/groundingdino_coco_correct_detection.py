# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T

# from torchvision.ops import box_convert

# from interpretation.submodular_detection import DetectionSubModularExplanation

# from tqdm import tqdm
# from utils import COCO_TEXT_PROMPT, coco_classes, coco_classes_grounding_idx, mkdir

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
#     # general
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Datasets.')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_single_detections.json',
#                         help='Datasets.')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='')
#     parser.add_argument('--begin', 
#                         type=int, default=0,
#                         help='')
#     parser.add_argument('--end', 
#                         type=int, default=-1,
#                         help='')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-correctly_again/',
#                         help='output directory to save results')
#     args = parser.parse_args()
#     return args

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler = 20.0) 
#         slic.iterate(20)     # The number of iterations, the larger the better the effect
#         label_slic = slic.getLabels()        # Get superpixel label
#         number_slic = slic.getNumberOfSuperpixels()  # Get the number of superpixels

#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:,:, np.newaxis].astype(int)
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image,10)  # The input image size must be the same as the initialization shape and the number of iterations is 10
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()

#         for i in range(number_seeds):
#             img_copp = (label_slic == i)[:,:, np.newaxis].astype(int)
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image):
#     """
#     Input:
#         image: An image read by opencv [w,h,c]
#     Output:
#         image: After preproccessing, is a tensor [c,w,h]
#     """
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     return image_transformed

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, 
#                  detection_model,
#                  device = "cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model
#         self.device = device
        
#         self.caption = None
    
#     def forward(self, images, h, w):
#         """_summary_

#         Args:
#             images (tensor): torch.Size([batch, 3, 773, 1332])
#         """
#         batch = images.shape[0]
#         captions = [self.caption for i in range(batch)]
        
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
            
#         prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (batch, nq, 256)
#         prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (batch, nq, 4)

#         boxes = prediction_boxes * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
#         return xyxy, prediction_logits 
    
# def main(args):
#     # model init
#     # Load the model
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")

#     detection_model = GroundingDino_Adaptation(model)
#     caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
#     detection_model.caption = caption
#     print("Load Grounding DINO model!")
    
#     # Submodular
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         transform_vision_data,
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device="cuda:2",
#         batch_size=20
#     )
    
#     # Read datasets
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
        
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))  
    
#     mkdir(save_dir)
    
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
    
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
    
#     end = args.end
#     if end == -1:
#         end = None
#     select_infos = val_file["case1"][args.begin : end]
#     id = args.begin + 1
    
#     for info in tqdm(select_infos):
#         if os.path.exists(
#             os.path.join(save_json_root_path, info["file_name"].replace(".jpg", "_{}.json".format(id)))
#         ):
#             id += 1
#             continue

#         target_class = coco_classes_grounding_idx[info["category"]]
#         target_box = info["bbox"]
#         image_path = os.path.join(args.Datasets, info["file_name"])
        
#         image = cv2.imread(image_path)
        
#         # Sub-region division
#         image_proccess = transform_vision_data(image)
#         image_seg = cv2.resize(image, image_proccess.shape[1:][::-1])

#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)

#         V_set = SubRegionDivision(image_seg, region_size = region_size)
        
#         S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
        
#         # Save npy file
#         np.save(
#             os.path.join(save_npy_root_path, info["file_name"].replace(".jpg", "_{}.npy".format(id))),
#             np.array(S_set)
#         )
        
#         # Save json file
#         with open(
#             os.path.join(save_json_root_path, info["file_name"].replace(".jpg", "_{}.json".format(id))), "w") as f:
#             f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
         
#         id += 1
#     return

# if __name__ == "__main__":
#     args = parse_args()
    
#     main(args)

# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# # Map COCO classes to indices based on their position in COCO_CLASSES
# coco_classes_grounding_idx = {cls: [i] for i, cls in enumerate(COCO_CLASSES)}

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_correct_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=500,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-correctly/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     print(f"Transformed image shape: {image_transformed.shape}")
#     return image_transformed

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         print(f"Processing batch of {batch} images with shape {images.shape}, h={h}, w={w}")
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].cpu().sigmoid()
#         prediction_boxes = outputs["pred_boxes"].cpu()
#         boxes = prediction_boxes * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits 

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
#     caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
#     detection_model.caption = caption
#     print("Loaded Grounding DINO model on", args.device)
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         transform_vision_data,
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=10
#     )
    
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     image_files = glob.glob(os.path.join(args.Datasets, "*.jpg"))
#     if not image_files:
#         print(f"Error: No .jpg files found in {args.Datasets}")
#         return
    
#     image_ids = []
#     for img_file in image_files:
#         filename = os.path.basename(img_file)
#         try:
#             img_id = int(filename.split('.')[0])
#             image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid filename format for {filename}, skipping")
#             continue
    
#     json_filenames = [info["file_name"] for info in val_file["case1"]]
#     json_image_ids = [int(f.split('.')[0]) for f in json_filenames]
#     valid_ids = [img_id for img_id in image_ids if img_id in json_image_ids]
#     if not valid_ids:
#         print(f"Error: No images in {args.Datasets} have entries in {args.eval_list}")
#         return
    
#     num_images = min(args.num_images, len(valid_ids))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
#     selected_ids = np.random.choice(valid_ids, size=num_images, replace=False).tolist()
#     select_infos = [info for info in val_file["case1"] if int(info["file_name"].split('.')[0]) in selected_ids]
#     if len(select_infos) != num_images:
#         print(f"Warning: Selected {len(select_infos)} images instead of {num_images}")
    
#     print(f"Selected {len(select_infos)} images from {args.Datasets} with valid JSON entries")
    
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, filename.replace(".jpg", f"_{id}.json"))
#         if os.path.exists(save_json_path):
#             id += 1
#             continue

#         target_class = coco_classes_grounding_idx[info["category"]]
#         target_box = info["bbox"]
#         image_path = os.path.join(args.Datasets, filename)
        
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         torch.cuda.empty_cache()
        
#         image_proccess = transform_vision_data(image)
#         image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))  # (width, height) = (W, H)
#         if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#             print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#             continue
        
#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#         try:
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#         except ValueError as e:
#             print(f"Error in SubRegionDivision for {filename}: {e}")
#             continue
        
#         try:
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             continue
        
#         np.save(
#             os.path.join(save_npy_root_path, filename.replace(".jpg", f"_{id}.npy")),
#             np.array(S_set)
#         )
        
#         with open(save_json_path, "w") as f:
#             json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
#         id += 1
    
#     print(f"Processed {id-1} images, results saved in {save_dir}")
#     return

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
    

# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# # Map COCO category IDs to COCO_CLASSES indices
# COCO_ID_TO_INDEX = {
#     1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
#     14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
#     24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
#     37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
#     48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
#     58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
#     72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
#     82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
# }

# # Map COCO class names to COCO_CLASSES indices
# COCO_NAME_TO_INDEX = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# # Visualization functions
# def add_value(S_set, json_file):
#     single_mask = np.zeros_like(S_set[0], dtype=np.float16)
#     value_list_1 = np.array(json_file["smdl_score"])
#     value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
#     value_list = value_list_1 - value_list_2
    
#     values = []
#     value = 0
#     for smdl_single_mask, smdl_value in zip(S_set, value_list):
#         value = value - abs(smdl_value)
#         single_mask[smdl_single_mask==1] = value
#         values.append(value)
    
#     attribution_map = single_mask - single_mask.min()
#     attribution_map = attribution_map / (attribution_map.max() + 1e-6)
#     return attribution_map, np.array(values)

# def gen_cam(image_path, mask):
#     w = mask.shape[1]
#     h = mask.shape[0]
#     image = cv2.resize(cv2.imread(image_path), (w, h))
#     mask = cv2.resize(mask, (int(w/20), int(h/20)))
#     mask = cv2.resize(mask, (w, h))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
#     heatmap = np.float32(heatmap)
#     cam = 0.5 * heatmap + 0.5 * np.float32(image)
#     return cam.astype(np.uint8), heatmap.astype(np.uint8)

# def norm_image(image):
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= (np.max(image) + 1e-6)
#     image *= 255.
#     return np.uint8(image)

# def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
#     boxes = torch.tensor(boxes, dtype=torch.float32)
#     class_ids = np.zeros(len(boxes), dtype=int)
#     h, w, _ = image.shape
#     boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
#     boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
#     xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
#     bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     annotated_frame = image
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
#     return annotated_frame

# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     x = [0.0] + saved_json_file["region_area"]
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     ax3.set_xlim((0, 1))
#     ax3.set_ylim((0, 1))
#     yticks = ax3.get_yticks()
#     yticks = yticks[yticks != 0]
#     ax3.set_yticks(yticks)
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
#     ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)
#     ax3.set_facecolor('white')
#     ax3.spines['bottom'].set_color('black')
#     ax3.spines['bottom'].set_linewidth(2.0)
#     ax3.spines['top'].set_color('none')
#     ax3.spines['left'].set_color('black')
#     ax3.spines['left'].set_linewidth(2.0)
#     ax3.spines['right'].set_color('none')
#     ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#     ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#     ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

#     kernel = np.ones((10, 10), dtype=np.uint8)
#     if mode == "insertion":
#         mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#     elif mode == "deletion":
#         mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

#     if ours_best_index != 0:
#         dilate = cv2.dilate(mask, kernel, iterations=3)
#         edge = dilate - mask
#     else:
#         edge = np.zeros_like(mask)

#     image_debug = image_baseline.copy()
#     image_debug[mask > 0] = image_debug[mask > 0] * 0.3
#     if ours_best_index != 0:
#         image_debug[edge > 0] = np.array([0, 0, 255])

#     if mode == "insertion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
#         color = (255, 69, 0)
#     elif mode == "deletion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
#         color = (30, 144, 255)

#     image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#     image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#     ax2.imshow(image_debug[..., ::-1])

#     auc = metrics.auc(x, curve_score)
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} {auc:.4f}", fontsize=54)
    
#     return fig

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_single_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=100,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-correctly/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image, device='cuda:2'):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     # print(f"Transformed image shape: {image_transformed.shape}, device: {image_transformed.device}")
#     image_transformed = image_transformed.to(device)
#     # print(f"Transformed image moved to device: {image_transformed.device}")
#     return image_transformed

# def convert_bbox_to_xyxy(bbox, image_shape):
#     """Convert [x, y, w, h] to [x1, y1, x2, y2] and clamp to image boundaries."""
#     x, y, w, h = bbox
#     x1 = x
#     y1 = y
#     x2 = x + w
#     y2 = y + h
#     img_h, img_w = image_shape[:2]
#     x1 = max(0, min(x1, img_w))
#     y1 = max(0, min(y1, img_h))
#     x2 = max(0, min(x2, img_w))
#     y2 = max(0, min(y2, img_h))
#     # print(f"Converted bbox {bbox} to xyxy: [{x1}, {y1}, {x2}, {y2}]")
#     return [x1, y1, x2, y2]

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         # print(f"Processing batch of {batch} images with shape {images.shape}, h={h}, w={w}, device: {images.device}")
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
#         prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
#         positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
#         if positive_map is not None:
#             print(f"positive_map shape: {positive_map.shape}")
#         boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits, positive_map

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
#     caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
#     detection_model.caption = caption
#     # print("Loaded Grounding DINO model on", args.device)
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         lambda x: transform_vision_data(x, device=args.device),
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=4  # Reduced to avoid OOM
#     )
    
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     print("First 5 JSON entries:")
#     for info in val_file["case1"][:5]:
#         print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Category ID: {info.get('category_id', 'N/A')}, Bbox: {info.get('bbox')}")
    
#     image_files = glob.glob(os.path.join(args.Datasets, "*.jpg"))
#     if not image_files:
#         print(f"Error: No .jpg files found in {args.Datasets}")
#         return
    
#     image_ids = []
#     for img_file in image_files:
#         filename = os.path.basename(img_file)
#         try:
#             img_id = int(filename.split('.')[0])
#             image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid filename format for {filename}, skipping")
#             continue
    
#     json_filenames = [info["file_name"] for info in val_file["case1"]]
#     json_image_ids = []
#     for f in json_filenames:
#         try:
#             img_id = int(f.split('.')[0])
#             json_image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid JSON filename format for {f}, skipping")
#             continue
    
#     valid_ids = [img_id for img_id in image_ids if img_id in json_image_ids]
#     if not valid_ids:
#         print(f"Error: No images in {args.Datasets} have entries in {args.eval_list}")
#         return
    
#     num_images = min(args.num_images, len(valid_ids))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
#     selected_ids = np.random.choice(valid_ids, size=num_images, replace=False).tolist()
#     select_infos = []
#     for info in val_file["case1"]:
#         try:
#             img_id = int(info["file_name"].split('.')[0])
#             if img_id in selected_ids:
#                 select_infos.append(info)
#                 if len(select_infos) >= num_images:
#                     break
#         except ValueError:
#             continue
#     print(f"Selected {len(select_infos)}/{num_images} images from {args.Datasets} with valid JSON entries")
    
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
#     save_vis_root_path = os.path.join(save_dir, "visualization")
#     mkdir(save_vis_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, f"{filename.replace('.jpg', '')}_{id}.json")
#         save_npy_path = os.path.join(save_npy_root_path, f"{filename.replace('.jpg', '')}_{id}.npy")
#         save_vis_path = os.path.join(save_vis_root_path, f"{filename.replace('.jpg', '')}_{id}.png")
#         if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
#             id += 1
#             continue

#         if "category" not in info:
#             print(f"Warning: Category not found in JSON for {filename}, skipping")
#             continue
#         category = info["category"]
#         if category not in COCO_NAME_TO_INDEX:
#             print(f"Warning: Category {category} not found in COCO_CLASSES for {filename}, skipping")
#             continue
#         target_class = COCO_NAME_TO_INDEX[category]
        
#         if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
#             print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
#             continue
        
#         image_path = os.path.join(args.Datasets, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
#         target_box = convert_bbox_to_xyxy(info["bbox"], image.shape)
        
#         class_name = info.get("category", "unknown")
        
#         torch.cuda.empty_cache()
        
#         image_proccess = transform_vision_data(image, device=args.device)
#         image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
#         if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#             print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#             continue
        
#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#         try:
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#         except ValueError as e:
#             print(f"Error in SubRegionDivision for {filename}: {e}")
#             continue
        
#         try:
#             # print(f"Processing {filename} with target_class: {target_class}, target_box: {target_box}, class_name: {class_name}")
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             continue
        
#         # Save npy and json
#         np.save(save_npy_path, np.array(S_set))
#         with open(save_json_path, "w") as f:
#             json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
#         # Generate and save visualization
#         try:
#             attribution_map, _ = add_value(S_set, saved_json_file)
#             vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
#             vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
#             vis_saliency_map_w_box = annotate_with_grounding_dino(
#                 vis_saliency_map,
#                 np.array([saved_json_file["target_box"]]),
#                 [f"{class_name}: {saved_json_file['insertion_cls'][-1] if saved_json_file['insertion_cls'] else 0:.2f}"]
#             )
#             fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
#             fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
#             plt.close(fig)
#             print(f"Saved visualization for {filename} at {save_vis_path}")
#         except Exception as e:
#             print(f"Error in visualization for {filename}: {e}")
#             continue
        
#         id += 1
#         torch.cuda.empty_cache()  # Clear after each image
    
#     print(f"Processed {id-1} images, results saved in {save_dir}")
#     return

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# # Map COCO category IDs to COCO_CLASSES indices
# COCO_ID_TO_INDEX = {
#     1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
#     14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
#     24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
#     37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
#     48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
#     58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
#     72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
#     82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
# }

# # Map COCO class names to COCO_CLASSES indices
# COCO_NAME_TO_INDEX = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# # Visualization functions
# def add_value(S_set, json_file):
#     single_mask = np.zeros_like(S_set[0], dtype=np.float16)
#     value_list_1 = np.array(json_file["smdl_score"])
#     value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
#     value_list = value_list_1 - value_list_2
    
#     values = []
#     value = 0
#     for smdl_single_mask, smdl_value in zip(S_set, value_list):
#         value = value - abs(smdl_value)
#         single_mask[smdl_single_mask==1] = value
#         values.append(value)
    
#     attribution_map = single_mask - single_mask.min()
#     attribution_map = attribution_map / (attribution_map.max() + 1e-6)
#     return attribution_map, np.array(values)

# def gen_cam(image_path, mask):
#     w = mask.shape[1]
#     h = mask.shape[0]
#     image = cv2.resize(cv2.imread(image_path), (w, h))
#     mask = cv2.resize(mask, (int(w/20), int(h/20)))
#     mask = cv2.resize(mask, (w, h))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
#     heatmap = np.float32(heatmap)
#     cam = 0.5 * heatmap + 0.5 * np.float32(image)
#     return cam.astype(np.uint8), heatmap.astype(np.uint8)

# def norm_image(image):
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= (np.max(image) + 1e-6)
#     image *= 255.
#     return np.uint8(image)

# def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
#     boxes = torch.tensor(boxes, dtype=torch.float32)
#     class_ids = np.zeros(len(boxes), dtype=int)
#     h, w, _ = image.shape
#     boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
#     boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
#     xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
#     bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     annotated_frame = image
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
#     return annotated_frame

# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     x = [0.0] + saved_json_file["region_area"]
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     ax3.set_xlim((0, 1))
#     ax3.set_ylim((0, 1))
#     yticks = ax3.get_yticks()
#     yticks = yticks[yticks != 0]
#     ax3.set_yticks(yticks)
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
#     ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)
#     ax3.set_facecolor('white')
#     ax3.spines['bottom'].set_color('black')
#     ax3.spines['bottom'].set_linewidth(2.0)
#     ax3.spines['top'].set_color('none')
#     ax3.spines['left'].set_color('black')
#     ax3.spines['left'].set_linewidth(2.0)
#     ax3.spines['right'].set_color('none')
#     ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#     ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#     ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

#     kernel = np.ones((10, 10), dtype=np.uint8)
#     if mode == "insertion":
#         mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#     elif mode == "deletion":
#         mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

#     if ours_best_index != 0:
#         dilate = cv2.dilate(mask, kernel, iterations=3)
#         edge = dilate - mask
#     else:
#         edge = np.zeros_like(mask)

#     image_debug = image_baseline.copy()
#     image_debug[mask > 0] = image_debug[mask > 0] * 0.3
#     if ours_best_index != 0:
#         image_debug[edge > 0] = np.array([0, 0, 255])

#     if mode == "insertion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
#         color = (255, 69, 0)
#     elif mode == "deletion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
#         color = (30, 144, 255)

#     image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#     image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#     ax2.imshow(image_debug[..., ::-1])

#     auc = metrics.auc(x, curve_score)
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} {auc:.4f}", fontsize=54)
    
#     return fig

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_correct_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=100,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-correctly/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image, device='cuda:2'):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     # print(f"Transformed image shape: {image_transformed.shape}, device: {image_transformed.device}")
#     image_transformed = image_transformed.to(device)
#     # print(f"Transformed image moved to device: {image_transformed.device}")
#     return image_transformed

# def validate_xyxy_bbox(bbox, image_shape):
#     """Validate and clamp xyxy bbox coordinates to image boundaries."""
#     x1, y1, x2, y2 = bbox
#     img_h, img_w = image_shape[:2]
    
#     # Clamp coordinates to image boundaries
#     x1 = max(0, min(x1, img_w))
#     y1 = max(0, min(y1, img_h))
#     x2 = max(0, min(x2, img_w))
#     y2 = max(0, min(y2, img_h))
    
#     # Ensure x2 > x1 and y2 > y1
#     if x2 <= x1:
#         x2 = min(x1 + 1, img_w)
#     if y2 <= y1:
#         y2 = min(y1 + 1, img_h)
    
#     # print(f"Validated xyxy bbox: [{x1}, {y1}, {x2}, {y2}] for image shape {image_shape}")
#     return [x1, y1, x2, y2]

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         # print(f"Processing batch of {batch} images with shape {images.shape}, h={h}, w={w}, device: {images.device}")
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
#         prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
#         positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
#         if positive_map is not None:
#             print(f"positive_map shape: {positive_map.shape}")
#         boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits, positive_map

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
#     caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
#     detection_model.caption = caption
#     # print("Loaded Grounding DINO model on", args.device)
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         lambda x: transform_vision_data(x, device=args.device),
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=4  # Reduced to avoid OOM
#     )
    
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     print("First 5 JSON entries:")
#     for info in val_file["case1"][:5]:
#         print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Category ID: {info.get('category_id', 'N/A')}, Bbox: {info.get('bbox')}")
    
#     image_files = glob.glob(os.path.join(args.Datasets, "*.jpg"))
#     if not image_files:
#         print(f"Error: No .jpg files found in {args.Datasets}")
#         return
    
#     image_ids = []
#     for img_file in image_files:
#         filename = os.path.basename(img_file)
#         try:
#             img_id = int(filename.split('.')[0])
#             image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid filename format for {filename}, skipping")
#             continue
    
#     json_filenames = [info["file_name"] for info in val_file["case1"]]
#     json_image_ids = []
#     for f in json_filenames:
#         try:
#             img_id = int(f.split('.')[0])
#             json_image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid JSON filename format for {f}, skipping")
#             continue
    
#     valid_ids = [img_id for img_id in image_ids if img_id in json_image_ids]
#     if not valid_ids:
#         print(f"Error: No images in {args.Datasets} have entries in {args.eval_list}")
#         return
    
#     num_images = min(args.num_images, len(valid_ids))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
#     selected_ids = np.random.choice(valid_ids, size=num_images, replace=False).tolist()
#     select_infos = []
#     for info in val_file["case1"]:
#         try:
#             img_id = int(info["file_name"].split('.')[0])
#             if img_id in selected_ids:
#                 select_infos.append(info)
#                 if len(select_infos) >= num_images:
#                     break
#         except ValueError:
#             continue
#     print(f"Selected {len(select_infos)}/{num_images} images from {args.Datasets} with valid JSON entries")
    
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
#     save_vis_root_path = os.path.join(save_dir, "visualization")
#     mkdir(save_vis_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, f"{filename.replace('.jpg', '')}_{id}.json")
#         save_npy_path = os.path.join(save_npy_root_path, f"{filename.replace('.jpg', '')}_{id}.npy")
#         save_vis_path = os.path.join(save_vis_root_path, f"{filename.replace('.jpg', '')}_{id}.png")
#         if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
#             id += 1
#             continue

#         if "category" not in info:
#             print(f"Warning: Category not found in JSON for {filename}, skipping")
#             continue
#         category = info["category"]
#         if category not in COCO_NAME_TO_INDEX:
#             print(f"Warning: Category {category} not found in COCO_CLASSES for {filename}, skipping")
#             continue
#         target_class = COCO_NAME_TO_INDEX[category]
        
#         if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
#             print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
#             continue
        
#         image_path = os.path.join(args.Datasets, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Bbox is already in xyxy format, just validate and clamp to image boundaries
#         target_box = validate_xyxy_bbox(info["bbox"], image.shape)
        
#         class_name = info.get("category", "unknown")
        
#         torch.cuda.empty_cache()
        
#         image_proccess = transform_vision_data(image, device=args.device)
#         image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
#         if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#             print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#             continue
        
#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#         try:
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#         except ValueError as e:
#             print(f"Error in SubRegionDivision for {filename}: {e}")
#             continue
        
#         try:
#             # print(f"Processing {filename} with target_class: {target_class}, target_box: {target_box}, class_name: {class_name}")
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             continue
        
#         # Save npy and json
#         np.save(save_npy_path, np.array(S_set))
#         with open(save_json_path, "w") as f:
#             json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
#         # Generate and save visualization
#         try:
#             attribution_map, _ = add_value(S_set, saved_json_file)
#             vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
#             vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
#             vis_saliency_map_w_box = annotate_with_grounding_dino(
#                 vis_saliency_map,
#                 np.array([saved_json_file["target_box"]]),
#                 [f"{class_name}: {saved_json_file['insertion_cls'][-1] if saved_json_file['insertion_cls'] else 0:.2f}"]
#             )
#             fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
#             fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
#             plt.close(fig)
#             print(f"Saved visualization for {filename} at {save_vis_path}")
#         except Exception as e:
#             print(f"Error in visualization for {filename}: {e}")
#             continue
        
#         id += 1
#         torch.cuda.empty_cache()  # Clear after each image
    
#     print(f"Processed {id-1} images, results saved in {save_dir}")
#     return

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# COCO_TEXT_PROMPT = " . ".join(COCO_CLASSES) + " ."

# # Map COCO category IDs to COCO_CLASSES indices
# COCO_ID_TO_INDEX = {
#     1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
#     14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
#     24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
#     37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
#     48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
#     58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
#     72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
#     82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
# }

# # Map COCO class names to COCO_CLASSES indices
# COCO_NAME_TO_INDEX = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# # Debug function to print score and area information
# def debug_scores_and_areas(saved_json_file, filename):
#     """Debug function to print score and area information"""
#     print(f"\n=== Debug Information for {filename} ===")
#     print(f"Baseline score: {saved_json_file.get('baseline_score', 'N/A')}")
#     print(f"Original score: {saved_json_file.get('org_score', 'N/A')}")
    
#     insertion_scores = saved_json_file.get('insertion_score', [])
#     deletion_scores = saved_json_file.get('deletion_score', [])
#     region_areas = saved_json_file.get('region_area', [])
    
#     print(f"Number of insertion scores: {len(insertion_scores)}")
#     print(f"Number of deletion scores: {len(deletion_scores)}")
#     print(f"Number of region areas: {len(region_areas)}")
    
#     if insertion_scores:
#         print(f"Insertion scores (first 5): {[f'{s:.4f}' for s in insertion_scores[:5]]}")
#         print(f"Insertion scores (last 5): {[f'{s:.4f}' for s in insertion_scores[-5:]]}")
#         print(f"Insertion score range: {min(insertion_scores):.4f} to {max(insertion_scores):.4f}")
#         # Check monotonicity
#         increases = sum(1 for i in range(1, len(insertion_scores)) 
#                        if insertion_scores[i] >= insertion_scores[i-1])
#         print(f"Insertion non-decreasing steps: {increases}/{len(insertion_scores)-1}")
        
#         # Check for flat regions
#         tolerance = 1e-6
#         flat_steps = sum(1 for i in range(1, len(insertion_scores)) 
#                         if abs(insertion_scores[i] - insertion_scores[i-1]) < tolerance)
#         print(f"Flat steps (tolerance={tolerance}): {flat_steps}/{len(insertion_scores)-1}")
    
#     if region_areas:
#         print(f"Region areas (first 5): {[f'{a:.4f}' for a in region_areas[:5]]}")
#         print(f"Region areas (last 5): {[f'{a:.4f}' for a in region_areas[-5:]]}")
#         print(f"Region area range: {min(region_areas):.4f} to {max(region_areas):.4f}")
#         print(f"Final region area: {region_areas[-1]:.4f}")
        
#         # Check if areas are increasing (they should be cumulative)
#         increases = sum(1 for i in range(1, len(region_areas)) 
#                        if region_areas[i] >= region_areas[i-1])
#         print(f"Area non-decreasing steps: {increases}/{len(region_areas)-1}")
    
#     print("=" * 50)

# # Visualization functions
# def add_value(S_set, json_file):
#     single_mask = np.zeros_like(S_set[0], dtype=np.float16)
#     value_list_1 = np.array(json_file["smdl_score"])
#     value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
#     value_list = value_list_1 - value_list_2
    
#     values = []
#     value = 0
#     for smdl_single_mask, smdl_value in zip(S_set, value_list):
#         value = value - abs(smdl_value)
#         single_mask[smdl_single_mask==1] = value
#         values.append(value)
    
#     attribution_map = single_mask - single_mask.min()
#     attribution_map = attribution_map / (attribution_map.max() + 1e-6)
#     return attribution_map, np.array(values)

# def gen_cam(image_path, mask):
#     w = mask.shape[1]
#     h = mask.shape[0]
#     image = cv2.resize(cv2.imread(image_path), (w, h))
#     mask = cv2.resize(mask, (int(w/20), int(h/20)))
#     mask = cv2.resize(mask, (w, h))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
#     heatmap = np.float32(heatmap)
#     cam = 0.5 * heatmap + 0.5 * np.float32(image)
#     return cam.astype(np.uint8), heatmap.astype(np.uint8)

# def norm_image(image):
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= (np.max(image) + 1e-6)
#     image *= 255.
#     return np.uint8(image)

# def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
#     boxes = torch.tensor(boxes, dtype=torch.float32)
#     class_ids = np.zeros(len(boxes), dtype=int)
#     h, w, _ = image.shape
#     boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
#     boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
#     xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
#     bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     annotated_frame = image
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
#     return annotated_frame

# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     """Fixed visualization function with proper debugging and error handling"""
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

#     # FIX: Ensure x and curve_score have consistent lengths
#     x = [0.0] + saved_json_file["region_area"]
    
#     print(f"\n--- Visualization Debug for {mode} ---")
#     print(f"Original lengths - x: {len(x)}, curve_score: {len(curve_score)}")
    
#     if len(x) != len(curve_score):
#         print(f"Warning: Length mismatch - x: {len(x)}, curve_score: {len(curve_score)}")
#         min_len = min(len(x), len(curve_score))
#         x = x[:min_len]
#         curve_score = curve_score[:min_len]
#         print(f"Truncated to length: {min_len}")
    
#     if len(curve_score) == 0:
#         print("Error: Empty curve_score, cannot create visualization")
#         return None

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     # Ensure index is within bounds
#     ours_best_index = max(0, min(ours_best_index, len(curve_score) - 1))
#     print(f"Best index: {ours_best_index}, Score: {curve_score[ours_best_index]:.4f}")
    
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     # Attribution Map
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     # Searched Region visualization
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     # Plot curve
#     ax3.set_xlim((0, 1))
#     # Dynamic y-axis limits based on actual score range
#     if curve_score:
#         y_min = max(0, min(curve_score) - 0.1)
#         y_max = min(1, max(curve_score) + 0.1)
#     else:
#         y_min, y_max = 0, 1
#     ax3.set_ylim((y_min, y_max))
    
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
    
#     # Add debug information
#     print(f"Plotting curve with {len(x_)} points")
#     if x_:
#         print(f"X range: {min(x_):.4f} to {max(x_):.4f}")
#     if ours_y:
#         print(f"Y range: {min(ours_y):.4f} to {max(ours_y):.4f}")
    
#     if len(x_) > 0 and len(ours_y) > 0:
#         ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5, marker='o', markersize=4)
#         ax3.set_facecolor('white')
#         ax3.spines['bottom'].set_color('black')
#         ax3.spines['bottom'].set_linewidth(2.0)
#         ax3.spines['top'].set_color('none')
#         ax3.spines['left'].set_color('black')
#         ax3.spines['left'].set_linewidth(2.0)
#         ax3.spines['right'].set_color('none')
        
#         ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#         ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#         if ours_best_index < len(x_):
#             ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5, linestyle='--')

#     # FIX: Correct mask calculation for searched regions
#     kernel = np.ones((10, 10), dtype=np.uint8)
    
#     try:
#         if mode == "insertion":
#             # For insertion: show regions that HAVE been added up to the best index
#             if ours_best_index < len(S_set_add):
#                 mask = (S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#             else:
#                 mask = np.zeros_like(S_set_add[0].sum(-1), dtype='uint8')
#         elif mode == "deletion":
#             # For deletion: show regions that have NOT been removed
#             if ours_best_index < len(S_set_add):
#                 mask = 1 - (S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#             else:
#                 mask = np.ones_like(S_set_add[0].sum(-1), dtype='uint8')

#         # Create edge visualization
#         if ours_best_index > 0:
#             dilate = cv2.dilate(mask, kernel, iterations=3)
#             edge = dilate - mask
#         else:
#             edge = np.zeros_like(mask)

#         image_debug = image_baseline.copy()
#         # Darken unselected regions
#         image_debug[mask == 0] = (image_debug[mask == 0] * 0.3).astype(np.uint8)
#         # Highlight edges in red
#         if ours_best_index > 0:
#             image_debug[edge > 0] = np.array([0, 0, 255])

#         # Get bounding box information
#         if mode == "insertion":
#             if ours_best_index > 0 and "insertion_box" in saved_json_file and ours_best_index <= len(saved_json_file["insertion_box"]):
#                 target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#                 cls_score = saved_json_file["insertion_cls"][ours_best_index - 1] if "insertion_cls" in saved_json_file else 0.0
#             else:
#                 target_box = saved_json_file.get("target_box", [0, 0, 100, 100])
#                 cls_score = saved_json_file.get("org_score", 0.0)
#             color = (255, 69, 0)
#         elif mode == "deletion":
#             if ours_best_index > 0 and "deletion_box" in saved_json_file and ours_best_index <= len(saved_json_file["deletion_box"]):
#                 target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#                 cls_score = saved_json_file["deletion_cls"][ours_best_index - 1] if "deletion_cls" in saved_json_file else 0.0
#             else:
#                 target_box = saved_json_file.get("target_box", [0, 0, 100, 100])
#                 cls_score = saved_json_file.get("org_score", 0.0)
#             color = (30, 144, 255)

#         image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#         image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#         ax2.imshow(image_debug[..., ::-1])
        
#     except Exception as e:
#         print(f"Error in mask visualization: {e}")
#         # Show original image if mask processing fails
#         ax2.imshow(image[..., ::-1])

#     # Calculate AUC
#     if len(x_) > 1:
#         try:
#             auc = metrics.auc(x_, ours_y)
#         except Exception as e:
#             print(f"Error calculating AUC: {e}")
#             auc = 0.0
#     else:
#         auc = 0.0
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} AUC: {auc:.4f}", fontsize=54)
    
#     print(f"--- End Visualization Debug ---\n")
    
#     return fig

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_single_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=100,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-correctly/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image, device='cuda:2'):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     image_transformed = image_transformed.to(device)
#     return image_transformed

# def convert_bbox_to_xyxy(bbox, image_shape):
#     """Convert [x, y, w, h] to [x1, y1, x2, y2] and clamp to image boundaries."""
#     x, y, w, h = bbox
#     x1 = x
#     y1 = y
#     x2 = x + w
#     y2 = y + h
#     img_h, img_w = image_shape[:2]
#     x1 = max(0, min(x1, img_w))
#     y1 = max(0, min(y1, img_h))
#     x2 = max(0, min(x2, img_w))
#     y2 = max(0, min(y2, img_h))
#     return [x1, y1, x2, y2]

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
#         prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
#         positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
#         boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits, positive_map

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
#     caption = preprocess_caption(caption=COCO_TEXT_PROMPT)
#     detection_model.caption = caption
#     print("Loaded Grounding DINO model on", args.device)
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         lambda x: transform_vision_data(x, device=args.device),
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=4  # Reduced to avoid OOM
#     )
    
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     print("First 5 JSON entries:")
#     for info in val_file["case1"][:5]:
#         print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Category ID: {info.get('category_id', 'N/A')}, Bbox: {info.get('bbox')}")
    
#     image_files = glob.glob(os.path.join(args.Datasets, "*.jpg"))
#     if not image_files:
#         print(f"Error: No .jpg files found in {args.Datasets}")
#         return
    
#     image_ids = []
#     for img_file in image_files:
#         filename = os.path.basename(img_file)
#         try:
#             img_id = int(filename.split('.')[0])
#             image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid filename format for {filename}, skipping")
#             continue
    
#     json_filenames = [info["file_name"] for info in val_file["case1"]]
#     json_image_ids = []
#     for f in json_filenames:
#         try:
#             img_id = int(f.split('.')[0])
#             json_image_ids.append(img_id)
#         except ValueError:
#             print(f"Warning: Invalid JSON filename format for {f}, skipping")
#             continue
    
#     valid_ids = [img_id for img_id in image_ids if img_id in json_image_ids]
#     if not valid_ids:
#         print(f"Error: No images in {args.Datasets} have entries in {args.eval_list}")
#         return
    
#     num_images = min(args.num_images, len(valid_ids))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
#     selected_ids = np.random.choice(valid_ids, size=num_images, replace=False).tolist()
#     select_infos = []
#     for info in val_file["case1"]:
#         try:
#             img_id = int(info["file_name"].split('.')[0])
#             if img_id in selected_ids:
#                 select_infos.append(info)
#                 if len(select_infos) >= num_images:
#                     break
#         except ValueError:
#             continue
#     print(f"Selected {len(select_infos)}/{num_images} images from {args.Datasets} with valid JSON entries")
    
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
#     save_vis_root_path = os.path.join(save_dir, "visualization")
#     mkdir(save_vis_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, f"{filename.replace('.jpg', '')}_{id}.json")
#         save_npy_path = os.path.join(save_npy_root_path, f"{filename.replace('.jpg', '')}_{id}.npy")
#         save_vis_path = os.path.join(save_vis_root_path, f"{filename.replace('.jpg', '')}_{id}.png")
        
#         if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
#             print(f"Files already exist for {filename}, skipping...")
#             id += 1
#             continue

#         if "category" not in info:
#             print(f"Warning: Category not found in JSON for {filename}, skipping")
#             continue
#         category = info["category"]
#         if category not in COCO_NAME_TO_INDEX:
#             print(f"Warning: Category {category} not found in COCO_CLASSES for {filename}, skipping")
#             continue
#         target_class = COCO_NAME_TO_INDEX[category]
        
#         if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
#             print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
#             continue
        
#         image_path = os.path.join(args.Datasets, filename)
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
#         target_box = convert_bbox_to_xyxy(info["bbox"], image.shape)
        
#         class_name = info.get("category", "unknown")
        
#         print(f"\nProcessing {filename} (ID: {id})")
#         print(f"Target class: {target_class} ({class_name})")
#         print(f"Target box: {target_box}")
#         print(f"Image shape: {image.shape}")
        
#         torch.cuda.empty_cache()
        
#         try:
#             image_proccess = transform_vision_data(image, device=args.device)
#             image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
#             if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#                 print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#                 continue
            
#             region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#             print(f"Region size for superpixels: {region_size}")
            
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#             print(f"Generated {len(V_set)} superpixel regions")
            
#         except ValueError as e:
#             print(f"Error in preprocessing for {filename}: {e}")
#             continue
#         except Exception as e:
#             print(f"Unexpected error in preprocessing for {filename}: {e}")
#             continue
        
#         try:
#             print(f"Running submodular explanation...")
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
            
#             # ADD DEBUG OUTPUT
#             debug_scores_and_areas(saved_json_file, filename)
            
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
        
#         try:
#             # Save npy and json
#             print("Saving results...")
#             np.save(save_npy_path, np.array(S_set))
#             with open(save_json_path, "w") as f:
#                 json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
            
#             # Generate and save visualization
#             print("Generating visualization...")
#             attribution_map, _ = add_value(S_set, saved_json_file)
#             vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
#             vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
            
#             # Add bounding box to attribution map
#             target_box_for_vis = saved_json_file.get("target_box", target_box)
#             insertion_cls_score = saved_json_file.get('insertion_cls', [0.0])[-1] if saved_json_file.get('insertion_cls') else 0.0
            
#             vis_saliency_map_w_box = annotate_with_grounding_dino(
#                 vis_saliency_map,
#                 np.array([target_box_for_vis]),
#                 [f"{class_name}: {insertion_cls_score:.2f}"]
#             )
            
#             # Create visualization
#             fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
            
#             if fig is not None:
#                 fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
#                 plt.close(fig)
#                 print(f"Saved visualization for {filename} at {save_vis_path}")
#             else:
#                 print(f"Failed to create visualization for {filename}")
                
#         except Exception as e:
#             print(f"Error in saving/visualization for {filename}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
        
#         id += 1
#         torch.cuda.empty_cache()
#         print(f"Successfully processed {filename}")
    
#     print(f"\nProcessed {id-1} images, results saved in {save_dir}")
#     return

# def validate_json_structure(saved_json_file, filename):
#     """Validate that the JSON structure contains expected fields"""
#     required_fields = ['baseline_score', 'org_score', 'insertion_score', 'region_area']
#     missing_fields = []
    
#     for field in required_fields:
#         if field not in saved_json_file:
#             missing_fields.append(field)
    
#     if missing_fields:
#         print(f"Warning: Missing fields in JSON for {filename}: {missing_fields}")
#         return False
    
#     # Check if lists have consistent lengths
#     insertion_score = saved_json_file.get('insertion_score', [])
#     region_area = saved_json_file.get('region_area', [])
    
#     if len(insertion_score) != len(region_area):
#         print(f"Warning: Length mismatch in {filename} - insertion_score: {len(insertion_score)}, region_area: {len(region_area)}")
#         return False
    
#     return True

# def create_simple_test_visualization(image_path, save_path):
#     """Create a simple test visualization to verify the pipeline works"""
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             return False
        
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
#         # Original image
#         axes[0].imshow(image[..., ::-1])
#         axes[0].set_title('Original Image')
#         axes[0].axis('off')
        
#         # Dummy attribution map
#         h, w = image.shape[:2]
#         dummy_attribution = np.random.rand(h, w)
#         axes[1].imshow(dummy_attribution, cmap='hot')
#         axes[1].set_title('Dummy Attribution')
#         axes[1].axis('off')
        
#         # Dummy curve
#         x = np.linspace(0, 1, 10)
#         y = np.random.rand(10)
#         axes[2].plot(x, y)
#         axes[2].set_title('Dummy Curve')
#         axes[2].set_xlabel('Percentage')
#         axes[2].set_ylabel('Score')
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=100, bbox_inches='tight')
#         plt.close(fig)
        
#         return True
#     except Exception as e:
#         print(f"Error in simple test visualization: {e}")
#         return False

# if __name__ == "__main__":
#     args = parse_args()
    
#     # Add some validation
#     if not os.path.exists(args.Datasets):
#         print(f"Error: Dataset directory {args.Datasets} does not exist")
#         exit(1)
        
#     if not os.path.exists(args.eval_list):
#         print(f"Error: Evaluation list {args.eval_list} does not exist")
#         exit(1)
    
#     # Check if CUDA is available
#     if not torch.cuda.is_available():
#         print("Warning: CUDA is not available, this will be very slow")
#         args.device = "cpu"
    
#     print(f"Using device: {args.device}")
#     print(f"Processing {args.num_images} images from {args.Datasets}")
#     print(f"Using superpixel algorithm: {args.superpixel_algorithm}")
#     print(f"Division number: {args.division_number}")
#     print(f"Lambda1: {args.lambda1}, Lambda2: {args.lambda2}")
    
#     try:
#         main(args)
#     except KeyboardInterrupt:
#         print("\nInterrupted by user")
#     except Exception as e:
#         print(f"Fatal error: {e}")
#         import traceback
#         traceback.print_exc()


# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# import re
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# # Map COCO category IDs to COCO_CLASSES indices
# COCO_ID_TO_INDEX = {
#     1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
#     14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
#     24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
#     37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
#     48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
#     58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
#     72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
#     82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
# }

# # Map COCO class names to COCO_CLASSES indices
# COCO_NAME_TO_INDEX = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# # Visualization functions
# def add_value(S_set, json_file):
#     single_mask = np.zeros_like(S_set[0], dtype=np.float16)
#     value_list_1 = np.array(json_file["smdl_score"])
#     value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
#     value_list = value_list_1 - value_list_2
    
#     values = []
#     value = 0
#     for smdl_single_mask, smdl_value in zip(S_set, value_list):
#         value = value - abs(smdl_value)
#         single_mask[smdl_single_mask==1] = value
#         values.append(value)
    
#     attribution_map = single_mask - single_mask.min()
#     attribution_map = attribution_map / (attribution_map.max() + 1e-6)
#     return attribution_map, np.array(values)

# def gen_cam(image_path, mask):
#     w = mask.shape[1]
#     h = mask.shape[0]
#     image = cv2.resize(cv2.imread(image_path), (w, h))
#     mask = cv2.resize(mask, (int(w/20), int(h/20)))
#     mask = cv2.resize(mask, (w, h))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
#     heatmap = np.float32(heatmap)
#     cam = 0.5 * heatmap + 0.5 * np.float32(image)
#     return cam.astype(np.uint8), heatmap.astype(np.uint8)

# def norm_image(image):
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= (np.max(image) + 1e-6)
#     image *= 255.
#     return np.uint8(image)

# def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
#     boxes = torch.tensor(boxes, dtype=torch.float32)
#     class_ids = np.zeros(len(boxes), dtype=int)
#     h, w, _ = image.shape
#     boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
#     boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
#     xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
#     bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     annotated_frame = image
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
#     return annotated_frame

# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     x = [0.0] + saved_json_file["region_area"]
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     ax3.set_xlim((0, 1))
#     ax3.set_ylim((0, 1))
#     yticks = ax3.get_yticks()
#     yticks = yticks[yticks != 0]
#     ax3.set_yticks(yticks)
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
#     ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)
#     ax3.set_facecolor('white')
#     ax3.spines['bottom'].set_color('black')
#     ax3.spines['bottom'].set_linewidth(2.0)
#     ax3.spines['top'].set_color('none')
#     ax3.spines['left'].set_color('black')
#     ax3.spines['left'].set_linewidth(2.0)
#     ax3.spines['right'].set_color('none')
#     ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#     ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#     ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

#     kernel = np.ones((10, 10), dtype=np.uint8)
#     if mode == "insertion":
#         mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#     elif mode == "deletion":
#         mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

#     if ours_best_index != 0:
#         dilate = cv2.dilate(mask, kernel, iterations=3)
#         edge = dilate - mask
#     else:
#         edge = np.zeros_like(mask)

#     image_debug = image_baseline.copy()
#     image_debug[mask > 0] = image_debug[mask > 0] * 0.3
#     if ours_best_index != 0:
#         image_debug[edge > 0] = np.array([0, 0, 255])

#     if mode == "insertion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
#         color = (255, 69, 0)
#     elif mode == "deletion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
#         color = (30, 144, 255)

#     image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#     image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#     ax2.imshow(image_debug[..., ::-1])

#     auc = metrics.auc(x, curve_score)
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} {auc:.4f}", fontsize=54)
    
#     return fig

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model on COCO Dataset')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_correct_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=100,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-coco/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def normalize_class_name(class_name: str) -> str:
#     """Normalize class names, ensuring non-empty output."""
#     norm_name = re.sub(r'[^a-zA-Z0-9\s]', '', class_name.lower().strip())
#     return norm_name if norm_name else class_name.lower().replace(' ', '_')

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image, device='cuda:2'):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     image_transformed = image_transformed.to(device)
#     return image_transformed

# def convert_bbox_to_xyxy(bbox, image_shape):
#     """Convert [x, y, w, h] to [x1, y1, x2, y2] and clamp to image boundaries."""
#     x, y, w, h = bbox
#     x1 = x
#     y1 = y
#     x2 = x + w
#     y2 = y + h
#     img_h, img_w = image_shape[:2]
#     x1 = max(0, min(x1, img_w))
#     y1 = max(0, min(y1, img_h))
#     x2 = max(0, min(x2, img_w))
#     y2 = max(0, min(y2, img_h))
#     return [x1, y1, x2, y2]

# def find_image_path(img_folder, filename):
#     """Helper function to find the correct image path for COCO dataset"""
#     base_filename = os.path.basename(filename)
    
#     # Strategy 1: Try original path from JSON
#     img_path = os.path.join(img_folder, filename)
#     if os.path.exists(img_path):
#         return img_path
    
#     # Strategy 2: Try just the basename in the main folder
#     img_path_base = os.path.join(img_folder, base_filename)
#     if os.path.exists(img_path_base):
#         return img_path_base
    
#     return None

# def build_class_mapping():
#     """Build a mapping from COCO category names to indices for the text prompt"""
#     # Normalize category names for text prompt
#     normalized_categories = [normalize_class_name(cat) for cat in COCO_CLASSES]
    
#     # Create text prompt
#     text_prompt = " . ".join(normalized_categories) + " ."
    
#     print(f"Using {len(COCO_CLASSES)} COCO categories")
#     print(f"Sample categories: {COCO_CLASSES[:10]}")
#     print(f"Text prompt length: {len(text_prompt)} characters")
    
#     return COCO_NAME_TO_INDEX, text_prompt, COCO_CLASSES

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
#         prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
#         positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
#         if positive_map is not None:
#             print(f"positive_map shape: {positive_map.shape}")
#         boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits, positive_map

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
    
#     # Load evaluation data
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     print("First 5 JSON entries:")
#     for info in val_file["case1"][:5]:
#         print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Category ID: {info.get('category_id', 'N/A')}, Bbox: {info.get('bbox')}")
    
#     # Build category mapping and text prompt
#     category_to_index, text_prompt, sorted_categories = build_class_mapping()
#     caption = preprocess_caption(text_prompt)
#     detection_model.caption = caption
#     print(f"Using text prompt with {len(sorted_categories)} categories")
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         lambda x: transform_vision_data(x, device=args.device),
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=4  # Reduced to avoid OOM
#     )
    
#     # Filter available images
#     available_images = []
#     for info in val_file["case1"]:
#         filename = info["file_name"]
#         image_path = find_image_path(args.Datasets, filename)
#         if image_path is not None:
#             available_images.append(info)
#         else:
#             print(f"Warning: Could not find image file for {filename}")
    
#     print(f"Found {len(available_images)} available images out of {len(val_file['case1'])} total")
    
#     if not available_images:
#         print(f"Error: No images found in {args.Datasets}")
#         return
    
#     # Select subset of images to process
#     num_images = min(args.num_images, len(available_images))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
    
#     selected_images = np.random.choice(len(available_images), size=num_images, replace=False)
#     select_infos = [available_images[i] for i in selected_images]
#     print(f"Selected {len(select_infos)} images for processing")
    
#     # Create output directories
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
#     save_vis_root_path = os.path.join(save_dir, "visualization")
#     mkdir(save_vis_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.json")
#         save_npy_path = os.path.join(save_npy_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.npy")
#         save_vis_path = os.path.join(save_vis_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.png")
        
#         # Skip if already processed
#         if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
#             id += 1
#             continue

#         if "category" not in info:
#             print(f"Warning: Category not found in JSON for {filename}, skipping")
#             continue
        
#         category = info["category"]
#         if category not in category_to_index:
#             print(f"Warning: Category {category} not found in category mapping for {filename}, skipping")
#             continue
        
#         target_class = category_to_index[category]
        
#         if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
#             print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
#             continue
        
#         # Find image path
#         image_path = find_image_path(args.Datasets, filename)
#         if image_path is None:
#             print(f"Failed to find image: {filename}")
#             continue
            
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Check if bbox is in xyxy or xywh format and convert accordingly
#         bbox = info["bbox"]
#         if len(bbox) == 4:
#             # Assume COCO format is typically xyxy, but check if conversion is needed
#             # If bbox values seem like width/height are too small, it might be xywh
#             x1, y1, x2_or_w, y2_or_h = bbox
#             if x2_or_w < x1 or y2_or_h < y1:  # Likely xywh format
#                 target_box = convert_bbox_to_xyxy(bbox, image.shape)
#             else:  # Likely already xyxy format
#                 # Validate and clamp to image boundaries
#                 target_box = [max(0, min(x1, image.shape[1])),
#                             max(0, min(y1, image.shape[0])),
#                             max(0, min(x2_or_w, image.shape[1])),
#                             max(0, min(y2_or_h, image.shape[0]))]
#         else:
#             print(f"Warning: Invalid bbox format for {filename}, skipping")
#             continue
        
#         class_name = info.get("category", "unknown")
        
#         torch.cuda.empty_cache()
        
#         image_proccess = transform_vision_data(image, device=args.device)
#         image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
#         if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#             print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#             continue
        
#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#         try:
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#         except ValueError as e:
#             print(f"Error in SubRegionDivision for {filename}: {e}")
#             continue
        
#         try:
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             continue
        
#         # Save npy and json
#         np.save(save_npy_path, np.array(S_set))
#         with open(save_json_path, "w") as f:
#             json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
#         # Generate and save visualization
#         try:
#             attribution_map, _ = add_value(S_set, saved_json_file)
#             vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
#             vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
#             vis_saliency_map_w_box = annotate_with_grounding_dino(
#                 vis_saliency_map,
#                 np.array([saved_json_file["target_box"]]),
#                 [f"{class_name}: {saved_json_file['insertion_cls'][-1] if saved_json_file['insertion_cls'] else 0:.2f}"]
#             )
#             fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
#             fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
#             plt.close(fig)
#             print(f"Saved visualization for {filename} at {save_vis_path}")
#         except Exception as e:
#             print(f"Error in visualization for {filename}: {e}")
#             continue
        
#         id += 1
#         torch.cuda.empty_cache()  # Clear after each image
    
#     print(f"Processed {id-1} images, results saved in {save_dir}")
#     return

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


# import os
# import json
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
# import supervision as sv
# from sklearn import metrics
# import argparse
# import cv2.ximgproc 
# import re
# plt.rc('font', family="Arial")

# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
# from torchvision.ops import box_convert
# from interpretation.submodular_detection import DetectionSubModularExplanation
# from tqdm import tqdm
# import glob

# # Suppress tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Define COCO classes
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

# # Map COCO category IDs to COCO_CLASSES indices
# COCO_ID_TO_INDEX = {
#     1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
#     14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
#     24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
#     37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
#     48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
#     58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
#     72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
#     82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
# }

# data_transform = T.Compose(
#     [
#         T.RandomResize([800], max_size=1333),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

# # Visualization functions
# def add_value(S_set, json_file):
#     single_mask = np.zeros_like(S_set[0], dtype=np.float16)
#     value_list_1 = np.array(json_file["smdl_score"])
#     value_list_2 = np.array([1 - json_file["org_score"] + json_file["baseline_score"]] + json_file["smdl_score"][:-1])
#     value_list = value_list_1 - value_list_2
    
#     values = []
#     value = 0
#     for smdl_single_mask, smdl_value in zip(S_set, value_list):
#         value = value - abs(smdl_value)
#         single_mask[smdl_single_mask==1] = value
#         values.append(value)
    
#     attribution_map = single_mask - single_mask.min()
#     attribution_map = attribution_map / (attribution_map.max() + 1e-6)
#     return attribution_map, np.array(values)

# def gen_cam(image_path, mask):
#     w = mask.shape[1]
#     h = mask.shape[0]
#     image = cv2.resize(cv2.imread(image_path), (w, h))
#     mask = cv2.resize(mask, (int(w/20), int(h/20)))
#     mask = cv2.resize(mask, (w, h))
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_COOL)
#     heatmap = np.float32(heatmap)
#     cam = 0.5 * heatmap + 0.5 * np.float32(image)
#     return cam.astype(np.uint8), heatmap.astype(np.uint8)

# def norm_image(image):
#     image = image.copy()
#     image -= np.max(np.min(image), 0)
#     image /= (np.max(image) + 1e-6)
#     image *= 255.
#     return np.uint8(image)

# def annotate_with_grounding_dino(image, boxes, phrases, color=(34, 139, 34)):
#     boxes = torch.tensor(boxes, dtype=torch.float32)
#     class_ids = np.zeros(len(boxes), dtype=int)
#     h, w, _ = image.shape
#     boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
#     boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
#     xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
#     detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)
#     bbox_annotator = sv.BoxAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
#     annotated_frame = image
#     annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
#     annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
#     return annotated_frame

# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     x = [0.0] + saved_json_file["region_area"]
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     ax3.set_xlim((0, 1))
#     ax3.set_ylim((0, 1))
#     yticks = ax3.get_yticks()
#     yticks = yticks[yticks != 0]
#     ax3.set_yticks(yticks)
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
#     ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5)
#     ax3.set_facecolor('white')
#     ax3.spines['bottom'].set_color('black')
#     ax3.spines['bottom'].set_linewidth(2.0)
#     ax3.spines['top'].set_color('none')
#     ax3.spines['left'].set_color('black')
#     ax3.spines['left'].set_linewidth(2.0)
#     ax3.spines['right'].set_color('none')
#     ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#     ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#     ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

#     kernel = np.ones((10, 10), dtype=np.uint8)
#     if mode == "insertion":
#         mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#     elif mode == "deletion":
#         mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

#     if ours_best_index != 0:
#         dilate = cv2.dilate(mask, kernel, iterations=3)
#         edge = dilate - mask
#     else:
#         edge = np.zeros_like(mask)

#     image_debug = image_baseline.copy()
#     image_debug[mask > 0] = image_debug[mask > 0] * 0.3
#     if ours_best_index != 0:
#         image_debug[edge > 0] = np.array([0, 0, 255])

#     if mode == "insertion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
#         color = (255, 69, 0)
#     elif mode == "deletion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
#         color = (30, 144, 255)

#     image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#     image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#     ax2.imshow(image_debug[..., ::-1])

#     auc = metrics.auc(x, curve_score)
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} {auc:.4f}", fontsize=54)
    
#     return fig

# def parse_args():
#     parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model on COCO Dataset')
#     parser.add_argument('--Datasets',
#                         type=str,
#                         default='datasets/val2017',
#                         help='Path to val2017 directory')
#     parser.add_argument('--eval-list',
#                         type=str,
#                         default='datasets/coco_groundingdino_correct_detections.json',
#                         help='Path to detection JSON file')
#     parser.add_argument('--superpixel-algorithm',
#                         type=str,
#                         default="slico",
#                         choices=["slico", "seeds"],
#                         help="Superpixel algorithm")
#     parser.add_argument('--lambda1', 
#                         type=float, default=1.,
#                         help='Lambda1 for submodular explanation')
#     parser.add_argument('--lambda2', 
#                         type=float, default=1.,
#                         help='Lambda2 for submodular explanation')
#     parser.add_argument('--division-number', 
#                         type=int, default=50,
#                         help='Number of superpixel regions')
#     parser.add_argument('--num-images', 
#                         type=int, default=100,
#                         help='Number of images to process')
#     parser.add_argument('--save-dir', 
#                         type=str, default='./submodular_results/grounding-dino-coco/',
#                         help='Output directory for results')
#     parser.add_argument('--device', 
#                         type=str, default="cuda:2",
#                         help='Device to run model (cuda:2)')
#     return parser.parse_args()

# def preprocess_caption(caption: str) -> str:
#     result = caption.lower().strip()
#     if result.endswith("."):
#         return result
#     return result + "."

# def normalize_class_name(class_name: str) -> str:
#     """Normalize class names, ensuring non-empty output."""
#     norm_name = re.sub(r'[^a-zA-Z0-9\s]', '', class_name.lower().strip())
#     return norm_name if norm_name else class_name.lower().replace(' ', '_')

# def SubRegionDivision(image, mode="slico", region_size=30):
#     element_sets_V = []
#     if mode == "slico":
#         slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0) 
#         slic.iterate(20)
#         label_slic = slic.getLabels()
#         number_slic = slic.getNumberOfSuperpixels()
#         print(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
#         for i in range(number_slic):
#             img_copp = (label_slic == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     elif mode == "seeds":
#         seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
#         seeds.iterate(image, 10)
#         label_seeds = seeds.getLabels()
#         number_seeds = seeds.getNumberOfSuperpixels()
#         print(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
#         for i in range(number_seeds):
#             img_copp = (label_seeds == i)[:, :, np.newaxis].astype(np.uint8)
#             if img_copp.shape[:2] != image.shape[:2]:
#                 raise ValueError(f"Superpixel mask shape {img_copp.shape} does not match image shape {image.shape}")
#             element_sets_V.append(img_copp)
#     return element_sets_V

# def transform_vision_data(image, device='cuda:2'):
#     image = Image.fromarray(image)
#     image_transformed, _ = data_transform(image, None)
#     image_transformed = image_transformed.to(device)
#     return image_transformed

# def convert_bbox_to_xyxy(bbox, image_shape):
#     """Convert [x, y, w, h] to [x1, y1, x2, y2] and clamp to image boundaries."""
#     x, y, w, h = bbox
#     x1 = x
#     y1 = y
#     x2 = x + w
#     y2 = y + h
#     img_h, img_w = image_shape[:2]
#     x1 = max(0, min(x1, img_w))
#     y1 = max(0, min(y1, img_h))
#     x2 = max(0, min(x2, img_w))
#     y2 = max(0, min(y2, img_h))
#     return [x1, y1, x2, y2]

# def find_image_path(img_folder, filename):
#     """Helper function to find the correct image path for COCO dataset"""
#     base_filename = os.path.basename(filename)
    
#     # Strategy 1: Try original path from JSON
#     img_path = os.path.join(img_folder, filename)
#     if os.path.exists(img_path):
#         return img_path
    
#     # Strategy 2: Try just the basename in the main folder
#     img_path_base = os.path.join(img_folder, base_filename)
#     if os.path.exists(img_path_base):
#         return img_path_base
    
#     return None

# def build_class_mapping(eval_list_data):
#     """Build a mapping from category names to indices for the text prompt - similar to iNaturalist approach"""
#     categories = set()
#     for info in eval_list_data["case1"]:
#         if "category" in info:
#             categories.add(info["category"])
    
#     # Sort categories for consistent ordering
#     sorted_categories = sorted(list(categories))
    
#     # Normalize category names for text prompt
#     normalized_categories = [normalize_class_name(cat) for cat in sorted_categories]
    
#     # Create text prompt
#     text_prompt = " . ".join(normalized_categories) + " ."
    
#     # Create mapping from original category names to indices
#     category_to_index = {cat: idx for idx, cat in enumerate(sorted_categories)}
    
#     print(f"Found {len(sorted_categories)} unique categories")
#     print(f"Sample categories: {sorted_categories[:10]}")
#     print(f"Text prompt length: {len(text_prompt)} characters")
    
#     return category_to_index, text_prompt, sorted_categories

# class GroundingDino_Adaptation(torch.nn.Module):
#     def __init__(self, detection_model, device="cuda:2"):
#         super().__init__()
#         self.detection_model = detection_model.to(device)
#         self.device = device
#         self.caption = None
    
#     def forward(self, images, h, w):
#         batch = images.shape[0]
#         captions = [self.caption for _ in range(batch)]
#         if torch.isnan(images).any() or torch.isinf(images).any():
#             raise ValueError("Input images contain NaN or Inf values")
#         images = images.to(self.device)
#         with torch.no_grad():
#             outputs = self.detection_model(images, captions=captions)
#         prediction_logits = outputs["pred_logits"].sigmoid()  # [batch, np, num_tokens]
#         prediction_boxes = outputs["pred_boxes"]  # [batch, np, 4]
#         positive_map = outputs.get("positive_map", None)  # [num_classes, num_tokens] or None
#         if positive_map is not None:
#             print(f"positive_map shape: {positive_map.shape}")
#         boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
#         return xyxy, prediction_logits, positive_map

# def mkdir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def main(args):
#     torch.cuda.set_device(args.device)
#     torch.cuda.empty_cache()
    
#     model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
#     detection_model = GroundingDino_Adaptation(model, device=args.device)
    
#     # Load evaluation data
#     with open(args.eval_list, 'r', encoding='utf-8') as f:
#         val_file = json.load(f)
    
#     print("First 5 JSON entries:")
#     for info in val_file["case1"][:5]:
#         print(f"File: {info.get('file_name')}, Category: {info.get('category')}, Category ID: {info.get('category_id', 'N/A')}, Bbox: {info.get('bbox')}")
    
#     # Build category mapping and text prompt - using same approach as iNaturalist
#     category_to_index, text_prompt, sorted_categories = build_class_mapping(val_file)
#     caption = preprocess_caption(text_prompt)
#     detection_model.caption = caption
#     print(f"Using text prompt with {len(sorted_categories)} categories")
    
#     smdl = DetectionSubModularExplanation(
#         detection_model,
#         lambda x: transform_vision_data(x, device=args.device),
#         lambda1=args.lambda1,
#         lambda2=args.lambda2,
#         device=args.device,
#         batch_size=4  # Reduced to avoid OOM
#     )
    
#     # Filter available images
#     available_images = []
#     for info in val_file["case1"]:
#         filename = info["file_name"]
#         image_path = find_image_path(args.Datasets, filename)
#         if image_path is not None:
#             available_images.append(info)
#         else:
#             print(f"Warning: Could not find image file for {filename}")
    
#     print(f"Found {len(available_images)} available images out of {len(val_file['case1'])} total")
    
#     if not available_images:
#         print(f"Error: No images found in {args.Datasets}")
#         return
    
#     # Select subset of images to process
#     num_images = min(args.num_images, len(available_images))
#     if num_images == 0:
#         print(f"Error: No valid images found")
#         return
    
#     selected_images = np.random.choice(len(available_images), size=num_images, replace=False)
#     select_infos = [available_images[i] for i in selected_images]
#     print(f"Selected {len(select_infos)} images for processing")
    
#     # Create output directories
#     mkdir(args.save_dir)
#     save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(
#         args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
#     mkdir(save_dir)
#     save_npy_root_path = os.path.join(save_dir, "npy")
#     mkdir(save_npy_root_path)
#     save_json_root_path = os.path.join(save_dir, "json")
#     mkdir(save_json_root_path)
#     save_vis_root_path = os.path.join(save_dir, "visualization")
#     mkdir(save_vis_root_path)
    
#     id = 1
#     for info in tqdm(select_infos, desc="Processing images"):
#         filename = info["file_name"]
#         save_json_path = os.path.join(save_json_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.json")
#         save_npy_path = os.path.join(save_npy_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.npy")
#         save_vis_path = os.path.join(save_vis_root_path, f"{os.path.splitext(os.path.basename(filename))[0]}_{id}.png")
        
#         # Skip if already processed
#         if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
#             id += 1
#             continue

#         if "category" not in info:
#             print(f"Warning: Category not found in JSON for {filename}, skipping")
#             continue
        
#         category = info["category"]
#         if category not in category_to_index:
#             print(f"Warning: Category {category} not found in category mapping for {filename}, skipping")
#             continue
        
#         target_class = category_to_index[category]
        
#         if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
#             print(f"Warning: Invalid or missing bbox in JSON for {filename}, skipping")
#             continue
        
#         # Find image path
#         image_path = find_image_path(args.Datasets, filename)
#         if image_path is None:
#             print(f"Failed to find image: {filename}")
#             continue
            
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to load image: {image_path}")
#             continue
        
#         # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] - consistent with iNaturalist approach
#         target_box = convert_bbox_to_xyxy(info["bbox"], image.shape)
        
#         class_name = info.get("category", "unknown")
        
#         torch.cuda.empty_cache()
        
#         image_proccess = transform_vision_data(image, device=args.device)
#         image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
#         if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
#             print(f"Warning: Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
#             continue
        
#         region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
#         try:
#             V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
#         except ValueError as e:
#             print(f"Error in SubRegionDivision for {filename}: {e}")
#             continue
        
#         try:
#             S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
#         except Exception as e:
#             print(f"Error in submodular explanation for {filename}: {e}")
#             continue
        
#         # Save npy and json
#         np.save(save_npy_path, np.array(S_set))
#         with open(save_json_path, "w") as f:
#             json.dump(saved_json_file, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        
#         # Generate and save visualization
#         try:
#             attribution_map, _ = add_value(S_set, saved_json_file)
#             vis_saliency_map, heatmap = gen_cam(image_path, norm_image(attribution_map[:, :, 0]))
#             vis_saliency_map = cv2.resize(vis_saliency_map, (image.shape[1], image.shape[0]))
#             vis_saliency_map_w_box = annotate_with_grounding_dino(
#                 vis_saliency_map,
#                 np.array([saved_json_file["target_box"]]),
#                 [f"{class_name}: {saved_json_file['insertion_cls'][-1] if saved_json_file['insertion_cls'] else 0:.2f}"]
#             )
#             fig = visualization(image, S_set, saved_json_file, vis_saliency_map_w_box, class_name, mode="insertion")
#             fig.savefig(save_vis_path, bbox_inches='tight', dpi=100)
#             plt.close(fig)
#             print(f"Saved visualization for {filename} at {save_vis_path}")
#         except Exception as e:
#             print(f"Error in visualization for {filename}: {e}")
#             continue
        
#         id += 1
#         torch.cuda.empty_cache()  # Clear after each image
    
#     print(f"Processed {id-1} images, results saved in {save_dir}")
#     return

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define COCO classes
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

# Map COCO category IDs to COCO_CLASSES indices
COCO_ID_TO_INDEX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
    24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
    37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
    58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
    72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

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
# def visualization(image, S_set, saved_json_file, vis_image, class_name, index=None, mode="insertion"):
#     S_set_add = S_set.copy()
#     S_set_add = np.array([S_set_add[0] - S_set_add[0]] + S_set_add)
#     image_baseline = cv2.resize(image, (S_set[0].shape[1], S_set[0].shape[0]))
    
#     if mode == "insertion":
#         curve_score = [saved_json_file["baseline_score"]] + saved_json_file["insertion_score"]
#         conf_score = [0.0] + saved_json_file["confidence_score"]  # Add confidence scores
#     elif mode == "deletion":
#         curve_score = [saved_json_file["org_score"]] + saved_json_file["deletion_score"]
#         conf_score = [0.0] + saved_json_file["confidence_score"]  # Add confidence scores

#     if index is None:
#         ours_best_index = np.argmax(curve_score) if mode == "insertion" else np.argmin(curve_score)
#     else:
#         ours_best_index = index
    
#     x = [0.0] + saved_json_file["region_area"]
#     i = len(x)
    
#     fig = plt.figure(figsize=(30, 8))
#     ax1 = fig.add_axes([0.05, 0.1, 0.3, 0.8])
#     ax2 = fig.add_axes([0.37, 0.1, 0.3, 0.8])
#     ax3 = fig.add_axes([0.75, 0.1, 0.25, 0.8])
    
#     ax1.spines["left"].set_visible(False)
#     ax1.spines["right"].set_visible(False)
#     ax1.spines["top"].set_visible(False)
#     ax1.spines["bottom"].set_visible(False)
#     ax1.xaxis.set_visible(False)
#     ax1.yaxis.set_visible(False)
#     ax1.set_title('Attribution Map', fontsize=54)
#     ax1.set_facecolor('white')
#     ax1.imshow(vis_image[..., ::-1].astype(np.uint8))
    
#     ax2.spines["left"].set_visible(False)
#     ax2.spines["right"].set_visible(False)
#     ax2.spines["top"].set_visible(False)
#     ax2.spines["bottom"].set_visible(False)
#     ax2.xaxis.set_visible(True)
#     ax2.yaxis.set_visible(False)
#     ax2.set_title('Searched Region', fontsize=54)
#     ax2.set_facecolor('white')
#     ax2.set_xlabel(f"Object Score: {curve_score[ours_best_index]:.2f}, Confidence: {conf_score[ours_best_index]:.2f}", fontsize=44)
#     ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#     ax3.set_xlim((0, 1))
#     ax3.set_ylim((0, 1))
#     yticks = ax3.get_yticks()
#     yticks = yticks[yticks != 0]
#     ax3.set_yticks(yticks)
#     ax3.set_ylabel('Object Score', fontsize=44)
#     ax3.set_xlabel('Percentage of image revealed' if mode == "insertion" else 'Percentage of image removed', fontsize=44)
#     ax3.tick_params(axis='both', which='major', labelsize=36)

#     curve_color = "#FF4500" if mode == "insertion" else "#1E90FF"
#     x_ = x[:i]
#     ours_y = curve_score[:i]
#     ax3.plot(x_, ours_y, color=curve_color, linewidth=3.5, label='Object Score')
#     ax3.plot(x_, conf_score[:i], color="#00FF00", linewidth=3.5, label='Confidence Score')  # Plot confidence
#     ax3.legend(fontsize=36)
#     ax3.set_facecolor('white')
#     ax3.spines['bottom'].set_color('black')
#     ax3.spines['bottom'].set_linewidth(2.0)
#     ax3.spines['top'].set_color('none')
#     ax3.spines['left'].set_color('black')
#     ax3.spines['left'].set_linewidth(2.0)
#     ax3.spines['right'].set_color('none')
#     ax3.scatter(x_[-1], ours_y[-1], color=curve_color, s=54)
#     ax3.scatter(x_[-1], conf_score[-1], color="#00FF00", s=54)  # Add confidence point
#     ax3.fill_between(x_, ours_y, color=curve_color, alpha=0.1)
#     ax3.axvline(x=x_[ours_best_index], color='red', linewidth=3.5)

#     kernel = np.ones((10, 10), dtype=np.uint8)
#     if mode == "insertion":
#         mask = (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')
#     elif mode == "deletion":
#         mask = 1 - (S_set_add.sum(0) - S_set_add[:ours_best_index + 1].sum(0)).sum(-1).astype('uint8')

#     if ours_best_index != 0:
#         dilate = cv2.dilate(mask, kernel, iterations=3)
#         edge = dilate - mask
#     else:
#         edge = np.zeros_like(mask)

#     image_debug = image_baseline.copy()
#     image_debug[mask > 0] = image_debug[mask > 0] * 0.3
#     if ours_best_index != 0:
#         image_debug[edge > 0] = np.array([0, 0, 255])

#     if mode == "insertion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["insertion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["insertion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["deletion_box"][-1] if saved_json_file["deletion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["deletion_cls"][-1] if saved_json_file["deletion_cls"] else 0.0
#         color = (255, 69, 0)
#     elif mode == "deletion":
#         if ours_best_index != 0:
#             target_box = saved_json_file["deletion_box"][ours_best_index - 1]
#             cls_score = saved_json_file["deletion_cls"][ours_best_index - 1]
#         else:
#             target_box = saved_json_file["insertion_box"][-1] if saved_json_file["insertion_box"] else [0, 0, 0, 0]
#             cls_score = saved_json_file["insertion_cls"][-1] if saved_json_file["insertion_cls"] else 0.0
#         color = (30, 144, 255)

#     image_debug = cv2.resize(image_debug, (image.shape[1], image.shape[0]))
#     image_debug = annotate_with_grounding_dino(image_debug, np.array([target_box]), [f"{class_name}: {cls_score:.2f}"], color)
#     ax2.imshow(image_debug[..., ::-1])

#     auc = metrics.auc(x, curve_score)
#     ax3.set_title(f"{'Insertion' if mode == 'insertion' else 'Deletion'} AUC: {auc:.4f}", fontsize=54)
    
#     return fig

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model on COCO Dataset')
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/transformed_val2017',
                        help='Path to val2017 directory')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_groundingdino_transformed_correct_detections.json',
                        help='Path to detection JSON file')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="seeds",
                        choices=["slico", "seeds"],
                        help="Superpixel algorithm")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='Lambda1 for submodular explanation')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='Lambda2 for submodular explanation')
    # parser.add_argument('--lambda3', 
    #                     type=float, default=1.,  # Add lambda3
    #                     help='Lambda3 for confidence score in submodular explanation')
    parser.add_argument('--division-number', 
                        type=int, default=50,
                        help='Number of superpixel regions')
    parser.add_argument('--num-images', 
                        type=int, default=150,
                        help='Number of images to process')
    parser.add_argument('--save-dir', 
                        type=str, default='./submodular_results/grounding-dino-coco_transformed_seeds/',
                        help='Output directory for results')
    parser.add_argument('--device', 
                        type=str, default="cuda:3",
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
        log.info(f"SLIC: Generated {number_slic} superpixels for image shape {image.shape}")
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
        log.info(f"SEEDS: Generated {number_seeds} superpixels for image shape {image.shape}")
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
    """Helper function to find the correct image path for COCO dataset"""
    base_filename = os.path.basename(filename)
    
    # Strategy 1: Try original path from JSON
    img_path = os.path.join(img_folder, filename)
    if os.path.exists(img_path):
        return img_path
    
    # Strategy 2: Try just the basename in the main folder
    img_path_base = os.path.join(img_folder, base_filename)
    if os.path.exists(img_path_base):
        return img_path_base
    
    return None

def delete_existing_files(save_npy_root_path, save_json_root_path, save_vis_root_path, base_filename):
    """Delete existing .npy, .json, and .png files for the given image."""
    for ext, path in [('.npy', save_npy_root_path), ('.json', save_json_root_path), ('.png', save_vis_root_path)]:
        pattern = os.path.join(path, f"{base_filename}_*{ext}")
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                log.info(f"Deleted existing file: {file}")
            except OSError as e:
                log.warning(f"Failed to delete file {file}: {e}")

def build_class_mapping(eval_list_data):
    """Build a mapping from category names to indices for the text prompt."""
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
    
    log.info(f"Found {len(sorted_categories)} unique categories")
    log.info(f"Sample categories: {sorted_categories[:10]}")
    log.info(f"Text prompt length: {len(text_prompt)} characters")
    
    return category_to_index, text_prompt, sorted_categories

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
            log.info(f"positive_map shape: {positive_map.shape}")
        boxes = prediction_boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        return xyxy, prediction_logits, positive_map

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    
    model = load_model("config/GroundingDINO_SwinT_OGC.py", "ckpt/groundingdino_swint_ogc.pth")
    detection_model = GroundingDino_Adaptation(model, device=args.device)
    
    # Load evaluation data
    with open(args.eval_list, 'r', encoding='utf-8') as f:
        val_file = json.load(f)
    
    log.info("First 5 JSON entries:")
    for info in val_file["case1"][:5]:
        log.info(f"File: {info.get('file_name')}, Category: {info.get('category')}, Bbox: {info.get('bbox')}")
    
    # Build category mapping and text prompt
    category_to_index, text_prompt, sorted_categories = build_class_mapping(val_file)
    caption = preprocess_caption(text_prompt)
    detection_model.caption = caption
    log.info(f"Using text prompt with {len(sorted_categories)} categories")
    
    smdl = DetectionSubModularExplanation(
        detection_model,
        lambda x: transform_vision_data(x, device=args.device),
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        # lambda3=args.lambda3,  # Pass lambda3
        device=args.device,
        batch_size=4
    )
    
    # Filter available images
    available_images = []
    for info in val_file["case1"]:
        filename = info["file_name"]
        image_path = find_image_path(args.Datasets, filename)
        if image_path is not None:
            available_images.append(info)
        else:
            log.warning(f"Could not find image file for {filename}")
    
    log.info(f"Found {len(available_images)} available images out of {len(val_file['case1'])} total")
    
    if not available_images:
        log.error(f"No images found in {args.Datasets}")
        return
    
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
    
    # Track processed images and valid images
    processed_image_ids = set()
    valid_images_processed = 0
    target_valid_images = args.num_images  # 100
    id_counter = 1
    indices = list(range(len(available_images)))
    np.random.shuffle(indices)  # Shuffle indices for random selection
    
    with tqdm(total=target_valid_images, desc="Processing images") as pbar:
        idx = 0
        while valid_images_processed < target_valid_images and idx < len(indices):
            info_idx = indices[idx]
            info = available_images[info_idx]
            filename = info["file_name"]
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            
            # Generate unique ID for this image
            save_json_path = os.path.join(save_json_root_path, f"{base_filename}_{id_counter}.json")
            save_npy_path = os.path.join(save_npy_root_path, f"{base_filename}_{id_counter}.npy")
            save_vis_path = os.path.join(save_vis_root_path, f"{base_filename}_{id_counter}.png")
            
            # Check if image was already processed
            if base_filename in processed_image_ids:
                idx += 1
                continue
            
            # Skip if already processed and files exist
            if os.path.exists(save_json_path) and os.path.exists(save_npy_path) and os.path.exists(save_vis_path):
                with open(save_json_path, 'r') as f:
                    saved_json_file = json.load(f)
                if "insertion_score" in saved_json_file and all(score >= 0 for score in saved_json_file["insertion_score"]):
                    valid_images_processed += 1
                    processed_image_ids.add(base_filename)
                    id_counter += 1
                    pbar.update(1)
                    log.info(f"Reused existing valid image: {filename}")
                    idx += 1
                    continue
                else:
                    # Delete existing files if insertion scores are invalid
                    delete_existing_files(save_npy_root_path, save_json_root_path, save_vis_root_path, base_filename)
            
            if "category" not in info:
                log.warning(f"Category not found in JSON for {filename}, skipping")
                idx += 1
                continue
            
            category = info["category"]
            if category not in category_to_index:
                log.warning(f"Category {category} not found in category mapping for {filename}, skipping")
                idx += 1
                continue
            
            target_class = category_to_index[category]
            
            if "bbox" not in info or not isinstance(info["bbox"], (list, tuple)) or len(info["bbox"]) != 4:
                log.warning(f"Invalid or missing bbox in JSON for {filename}, skipping")
                idx += 1
                continue
            
            # Find image path
            image_path = find_image_path(args.Datasets, filename)
            if image_path is None:
                log.warning(f"Failed to find image: {filename}")
                idx += 1
                continue
                
            image = cv2.imread(image_path)
            if image is None:
                log.warning(f"Failed to load image: {image_path}")
                idx += 1
                continue
            
            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            target_box = convert_bbox_to_xyxy(info["bbox"], image.shape)
            
            class_name = info.get("category", "unknown")
            
            torch.cuda.empty_cache()
            
            image_proccess = transform_vision_data(image, device=args.device)
            image_seg = cv2.resize(image, (image_proccess.shape[2], image_proccess.shape[1]))
            if image_seg.shape[:2] != (image_proccess.shape[1], image_proccess.shape[2]):
                log.warning(f"Resized image shape {image_seg.shape} does not match transformed shape {image_proccess.shape[1:]}")
                idx += 1
                continue
            
            region_size = int((image_seg.shape[0] * image_seg.shape[1] / args.division_number) ** 0.5)
            try:
                V_set = SubRegionDivision(image_seg, mode=args.superpixel_algorithm, region_size=region_size)
            except ValueError as e:
                log.warning(f"Error in SubRegionDivision for {filename}: {e}")
                idx += 1
                continue
            
            try:
                S_set, saved_json_file = smdl(image, image_proccess, V_set, target_class, target_box)
            except Exception as e:
                log.warning(f"Error in submodular explanation for {filename}: {e}")
                idx += 1
                continue
            
            # Check insertion scores
            if "insertion_score" not in saved_json_file or not all(score >= 0 for score in saved_json_file["insertion_score"]):
                log.warning(f"Image {filename} rejected: Insertion score < 0.01")
                delete_existing_files(save_npy_root_path, save_json_root_path, save_vis_root_path, base_filename)
                idx += 1
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
                log.info(f"Saved visualization for {filename} at {save_vis_path}")
            except Exception as e:
                log.warning(f"Error in visualization for {filename}: {e}")
                delete_existing_files(save_npy_root_path, save_json_root_path, save_vis_root_path, base_filename)
                idx += 1
                continue
            
            valid_images_processed += 1
            processed_image_ids.add(base_filename)
            id_counter += 1
            pbar.update(1)
            log.info(f"Processed valid image {valid_images_processed}/{target_valid_images}: {filename}")
            idx += 1
        
            if idx >= len(indices) and valid_images_processed < target_valid_images:
                log.warning(f"Exhausted available images ({len(indices)}) with only {valid_images_processed} valid images processed")
                break
    
    log.info(f"Processed {valid_images_processed} valid images, results saved in {save_dir}")
    if valid_images_processed < target_valid_images:
        log.warning(f"Could not find {target_valid_images} images with insertion scores >= 0.1. Processed {valid_images_processed} images.")
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
