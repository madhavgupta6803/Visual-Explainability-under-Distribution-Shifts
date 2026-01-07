# import numpy as np

# from tqdm import tqdm

# import torch

# import time

# class DetectionSubModularExplanation(object):
#     """
#     Instance-level interpretability of object detection 
#     based on submodular subset selection
#     """
#     def __init__(self, 
#                  detection_model,
#                  preproccessing_function,
#                  lambda1 = 1.0,
#                  lambda2 = 1.0,
#                  batch_size = 4,    # Suggestion: [2080Ti: 4], [3090: 16]
#                  mode = "object",   # object, iou, cls
#                  device = "cuda:2"):
#         """_summary_

#         Args:
#             detection_model (_type_): _description_
#             preproccessing_function (_type_): _description_
#             lambda1 (float, optional): _description_. Defaults to 1.0.
#             lambda2 (float, optional): _description_. Defaults to 1.0.
#             device (str, optional): _description_. Defaults to "cuda".
#         """
#         super(DetectionSubModularExplanation, self).__init__()
        
#         # Parameters of the submodular
#         self.detection_model = detection_model.to(device)
#         self.preproccessing_function = preproccessing_function
        
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
        
#         self.device = device
        
#         self.batch_size = batch_size
#         self.mode = mode
        
#     def save_file_init(self):
#         self.saved_json_file = {}
#         self.saved_json_file["insertion_score"] = []
#         self.saved_json_file["deletion_score"] = []
#         self.saved_json_file["smdl_score"] = []
#         self.saved_json_file["insertion_iou"] = []
#         self.saved_json_file["insertion_box"] = []
#         self.saved_json_file["insertion_cls"] = []
#         self.saved_json_file["deletion_iou"] = []
#         self.saved_json_file["deletion_box"] = []
#         self.saved_json_file["deletion_cls"] = []
#         self.saved_json_file["region_area"] = []
#         self.saved_json_file["target_box"] = []
#         self.saved_json_file["lambda1"] = self.lambda1
#         self.saved_json_file["lambda2"] = self.lambda2
#         self.saved_json_file["mode"] = self.mode
    
#     def process_in_batches(self, images, batch_size, detection_model, h, w):
#         all_bounding_boxes = []
#         all_logits = []

#         # 将输入图像拆分为 batch_size 批次
#         num_batches = (len(images) + batch_size - 1) // batch_size  # 计算需要的批次数

#         for i in range(num_batches):
#             # 获取当前批次的图像
#             batch_images = images[i * batch_size:(i + 1) * batch_size]

#             # 将当前批次传递到检测模型
#             bounding_boxes, logits = detection_model(batch_images, h, w)

#             # 将结果收集到列表中
#             all_bounding_boxes.append(bounding_boxes)
#             all_logits.append(logits)

#         # 将所有批次的结果拼接成一个完整的张量
#         all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0)
#         all_logits = torch.cat(all_logits, dim=0)

#         return all_bounding_boxes, all_logits
    
#     def calculate_iou(self, batched_boxes, target_box):
#         # batched_boxes: [batch, np, 4]
#         # target_box: [4]

#         # Separation coordinates
#         x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
#         tx1, ty1, tx2, ty2 = torch.tensor(target_box)

#         # Calculate intersection area
#         inter_x1 = torch.maximum(x1, tx1)
#         inter_y1 = torch.maximum(y1, ty1)
#         inter_x2 = torch.minimum(x2, tx2)
#         inter_y2 = torch.minimum(y2, ty2)

#         # 计算相交区域的面积
#         inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)

#         # Calculate the area of ​​the intersection
#         box_area = (x2 - x1) * (y2 - y1)
#         target_area = (tx2 - tx1) * (ty2 - ty1)

#         # Calculating IoU
#         union_area = box_area + target_area - inter_area
#         iou = inter_area / union_area

#         return iou
    
#     def generate_masked_input(self, alpha_batch):
#         alpha_batch = torch.tensor(alpha_batch)
#         alpha_batch = alpha_batch.permute(0, 3, 1, 2)   # [batch, 1, 773, 1332]
#         alpha_batch = alpha_batch.repeat(1, 3, 1, 1) # [batch, 3, 773, 1332]
        
#         source_image_process = self.source_image_proccess.unsqueeze(0).repeat(alpha_batch.size(0), 1, 1, 1)  # [batch, 3, 773, 1332]
        
#         return alpha_batch * source_image_process
        
    
#     def evaluation_maximun_sample(self, S_set):
#         # timer = time.time()
#         V_set_tem = np.array(self.V_set) # (100, 773, 1332, 1)
        
#         alpha_batch = (V_set_tem + self.refer_baseline[np.newaxis,...]).astype(np.uint8) # (100, 773, 1332, 1)
        
#         # print("Stage 1 time comsume: {}".format(time.time()-timer))
#         # timer = time.time()
        
#         batch_input_images = self.generate_masked_input(alpha_batch).to(self.device)
#         batch_input_images_reverse = self.generate_masked_input(1-alpha_batch).to(self.device)
        
#         # print("Stage 2 time comsume: {}".format(time.time()-timer))
#         # timer = time.time()
        
#         with torch.no_grad():
#             # Insertion
#             bounding_boxes, logits = self.process_in_batches(batch_input_images, self.batch_size, self.detection_model, self.h, self.w) # [batch, np, 4] [batch, np, 256]
            
#             # print("Stage 3.1 time comsume: {}".format(time.time()-timer))
#             # timer = time.time()
            
#             ious = self.calculate_iou(bounding_boxes, self.target_box)
#             if self.mode == "cls":
#                 ious_clip = (ious>0.5).int()
#             elif self.mode == "object":
#                 ious_clip = ious
            
#             cls_score = logits[:,:,self.target_label].max(dim=-1)[0]   # torch.Size([170, 900])
            
#             insertion_scores = (ious_clip * cls_score).max(dim=-1)[0]
            
#             # print("Stage 3 time comsume: {}".format(time.time()-timer))
#             # timer = time.time()
            
#             # Deletion
#             bounding_boxes_reverse, logits_reverse = self.process_in_batches(batch_input_images_reverse, self.batch_size, self.detection_model, self.h, self.w) # [batch, np, 4] [batch, np, 256]
            
#             ious_reverse = self.calculate_iou(bounding_boxes_reverse, self.target_box)
#             if self.mode == "cls":
#                 ious_reverse_clip = (ious_reverse>0.5).int()
#             elif self.mode == "object":
#                 ious_reverse_clip = ious_reverse
            
#             cls_score_reverse = logits_reverse[:,:,self.target_label].max(dim=-1)[0]   # torch.Size([170, 900])
            
#             deletion_scores = (ious_reverse_clip * cls_score_reverse).max(dim=-1)[0]
            
#             # print("Stage 4 time comsume: {}".format(time.time()-timer))
#             # timer = time.time()
            
#             #Overall submodular score
#             smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1-deletion_scores)
#             arg_max_index = smdl_scores.argmax().cpu().item()
            
#             # print("Stage 5 time comsume: {}".format(time.time()-timer))
#             # timer = time.time()
            
#             # Save intermediate results
#             insertion_boxer = bounding_boxes[arg_max_index].cpu().numpy()
#             insertion_box_id = (ious[arg_max_index] * cls_score[arg_max_index]).argmax().cpu().item()
#             insertion_box = insertion_boxer[insertion_box_id].astype(int).tolist()
#             insertion_iou = ious[arg_max_index][insertion_box_id].cpu().item()
#             insertion_cls = cls_score[arg_max_index][insertion_box_id].cpu().item()
#             self.saved_json_file["insertion_iou"].append(insertion_iou)
#             self.saved_json_file["insertion_box"].append(insertion_box)
#             self.saved_json_file["insertion_cls"].append(insertion_cls)
            
#             deletion_boxer = bounding_boxes_reverse[arg_max_index].cpu().numpy()
#             deletion_box_id = (ious_reverse[arg_max_index] * cls_score_reverse[arg_max_index]).argmax().cpu().item()
#             deletion_box = deletion_boxer[deletion_box_id].astype(int).tolist()
#             deletion_iou = ious_reverse[arg_max_index][deletion_box_id].cpu().item()
#             deletion_cls = cls_score_reverse[arg_max_index][deletion_box_id].cpu().item()
#             self.saved_json_file["deletion_iou"].append(deletion_iou)
#             self.saved_json_file["deletion_box"].append(deletion_box)
#             self.saved_json_file["deletion_cls"].append(deletion_cls)
            
#             # Update
#             S_set.append(self.V_set[arg_max_index])
#             self.refer_baseline = self.refer_baseline+self.V_set[arg_max_index]
#             del self.V_set[arg_max_index]
            
#             self.saved_json_file["region_area"].append(
#                 self.refer_baseline.sum() / self.region_area
#             )
            
#             self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].cpu().item())
#             self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].cpu().item())
#             self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())

#         return S_set
    
#     def get_merge_set(self):
#         # define a subset
#         S_set = []
#         self.refer_baseline = np.zeros_like(self.V_set[0])
        
#         for i in tqdm(range(self.saved_json_file["sub-region_number"])):
#             S_set = self.evaluation_maximun_sample(S_set)
        
#         self.saved_json_file["org_score"] = self.saved_json_file["insertion_score"][-1]
#         self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_score"][-1]
        
#         return S_set
    
#     def __call__(self, image, image_proccess, V_set, class_id, given_box):
#         """_summary_

#         Args:
#             image (cv2 format): (h, w, 3)
#             V_set (_type_): (n, h, w, 3)
#             class_id (List [int, ...]): which classes?
#             given_box (xyxy): which boxes?
#         """
#         self.save_file_init()
#         self.saved_json_file["target_box"] = given_box
#         self.saved_json_file["sub-region_number"] = len(V_set)
        
#         self.source_image = image
#         self.source_image_proccess = image_proccess # torch.Size([3, 773, 1332])
#         self.h, self.w, _ = self.source_image.shape
#         self.region_area = image_proccess.shape[1] * image_proccess.shape[2]
        
#         self.V_set = V_set.copy()
#         self.target_label = torch.tensor(class_id)
#         self.target_box = given_box
#         self.saved_json_file["target_label"] = class_id
        
#         Submodular_Subset = self.get_merge_set()
        
#         self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
#         self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
#         return Submodular_Subset, self.saved_json_file

# import numpy as np
# from tqdm import tqdm
# import torch
# import time

# class DetectionSubModularExplanation(object):
#     """
#     Instance-level interpretability of object detection 
#     based on submodular subset selection
#     """
#     def __init__(self, 
#                  detection_model,
#                  preproccessing_function,
#                  lambda1=1.0,
#                  lambda2=1.0,
#                  batch_size=4,  # Suggestion: [2080Ti: 4], [3090: 16]
#                  mode="object",  # object, iou, cls
#                  device="cuda:2"):
#         """_summary_

#         Args:
#             detection_model: Detection model instance
#             preproccessing_function: Function to preprocess input images
#             lambda1 (float): Weight for insertion score. Defaults to 1.0.
#             lambda2 (float): Weight for deletion score. Defaults to 1.0.
#             batch_size (int): Number of images to process in a batch
#             mode (str): Scoring mode ('object', 'iou', 'cls')
#             device (str): Device to run model (e.g., 'cuda:2')
#         """
#         super(DetectionSubModularExplanation, self).__init__()
        
#         self.detection_model = detection_model.to(device)
#         self.preproccessing_function = preproccessing_function
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.device = device
#         self.batch_size = batch_size
#         self.mode = mode
        
#     def save_file_init(self):
#         self.saved_json_file = {
#             "insertion_score": [],
#             "deletion_score": [],
#             "smdl_score": [],
#             "insertion_iou": [],
#             "insertion_box": [],
#             "insertion_cls": [],
#             "deletion_iou": [],
#             "deletion_box": [],
#             "deletion_cls": [],
#             "region_area": [],
#             "target_box": [],
#             "lambda1": self.lambda1,
#             "lambda2": self.lambda2,
#             "mode": self.mode
#         }
    
#     def process_in_batches(self, images, batch_size, detection_model, h, w):
#         all_bounding_boxes = []
#         all_logits = []
#         num_batches = (len(images) + batch_size - 1) // batch_size

#         for i in range(num_batches):
#             batch_images = images[i * batch_size:(i + 1) * batch_size]
#             print(f"Processing batch of {len(batch_images)} images with shape {batch_images.shape}, h={h}, w={w}")
#             bounding_boxes, logits = detection_model(batch_images, h, w)
#             all_bounding_boxes.append(bounding_boxes)
#             all_logits.append(logits)

#         all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0)
#         all_logits = torch.cat(all_logits, dim=0)
#         return all_bounding_boxes, all_logits
    
#     def calculate_iou(self, batched_boxes, target_box):
#         # batched_boxes: [batch, np, 4] in xyxy format
#         # target_box: [4] in xyxy format
#         x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
#         tx1, ty1, tx2, ty2 = torch.tensor(target_box, device=batched_boxes.device)

#         inter_x1 = torch.maximum(x1, tx1)
#         inter_y1 = torch.maximum(y1, ty1)
#         inter_x2 = torch.minimum(x2, tx2)
#         inter_y2 = torch.minimum(y2, ty2)

#         inter_area = torch.clamp((inter_x2 - inter_x1), min=0) * torch.clamp((inter_y2 - inter_y1), min=0)
#         box_area = (x2 - x1) * (y2 - y1)
#         target_area = (tx2 - tx1) * (ty2 - ty1)
#         union_area = box_area + target_area - inter_area
#         iou = inter_area / (union_area + 1e-6)  # Avoid division by zero
#         return iou
    
#     def generate_masked_input(self, alpha_batch):
#         alpha_batch = torch.tensor(alpha_batch, device=self.device)
#         alpha_batch = alpha_batch.permute(0, 3, 1, 2)  # [batch, 1, H, W]
#         alpha_batch = alpha_batch.repeat(1, 3, 1, 1)  # [batch, 3, H, W]
#         source_image_process = self.source_image_proccess.unsqueeze(0).repeat(alpha_batch.size(0), 1, 1, 1)
#         return alpha_batch * source_image_process
        
#     def evaluation_maximun_sample(self, S_set):
#         V_set_tem = np.array(self.V_set)  # [n, H, W, 1]
#         alpha_batch = (V_set_tem + self.refer_baseline[np.newaxis, ...]).astype(np.uint8)  # [n, H, W, 1]
        
#         batch_input_images = self.generate_masked_input(alpha_batch).to(self.device)
#         batch_input_images_reverse = self.generate_masked_input(1 - alpha_batch).to(self.device)
        
#         with torch.no_grad():
#             # Insertion
#             bounding_boxes, logits = self.process_in_batches(batch_input_images, self.batch_size, self.detection_model, self.h, self.w)
#             ious = self.calculate_iou(bounding_boxes, self.target_box)
#             cls_score = logits[:, :, self.target_label].max(dim=-1)[0]  # [batch]
#             print(f"Insertion IoU max: {ious.max().item():.4f}, Cls score max: {cls_score.max().item():.4f}")
            
#             if self.mode == "cls":
#                 ious_clip = (ious > 0.5).int()
#             elif self.mode == "object":
#                 ious_clip = torch.where(ious > 0, ious, torch.tensor(1e-6, device=ious.device))  # Avoid zero IoU
#             else:
#                 raise ValueError(f"Unsupported mode: {self.mode}")
            
#             insertion_scores = (ious_clip * cls_score).max(dim=-1)[0]
            
#             # Deletion
#             bounding_boxes_reverse, logits_reverse = self.process_in_batches(batch_input_images_reverse, self.batch_size, self.detection_model, self.h, self.w)
#             ious_reverse = self.calculate_iou(bounding_boxes_reverse, self.target_box)
#             cls_score_reverse = logits_reverse[:, :, self.target_label].max(dim=-1)[0]
#             print(f"Deletion IoU max: {ious_reverse.max().item():.4f}, Cls score reverse max: {cls_score_reverse.max().item():.4f}")
            
#             if self.mode == "cls":
#                 ious_reverse_clip = (ious_reverse > 0.5).int()
#             elif self.mode == "object":
#                 ious_reverse_clip = torch.where(ious_reverse > 0, ious_reverse, torch.tensor(1e-6, device=ious_reverse.device))
            
#             deletion_scores = (ious_reverse_clip * cls_score_reverse).max(dim=-1)[0]
            
#             # Submodular score
#             smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
#             arg_max_index = smdl_scores.argmax().cpu().item()
#             print(f"SMDL score max: {smdl_scores.max().item():.4f}, Selected index: {arg_max_index}")
            
#             # Save results
#             insertion_boxer = bounding_boxes[arg_max_index].cpu().numpy()
#             insertion_box_id = (ious[arg_max_index] * cls_score[arg_max_index]).argmax().cpu().item()
#             insertion_box = insertion_boxer[insertion_box_id].astype(int).tolist()
#             insertion_iou = ious[arg_max_index][insertion_box_id].cpu().item()
#             insertion_cls = cls_score[arg_max_index][insertion_box_id].cpu().item()
#             self.saved_json_file["insertion_iou"].append(insertion_iou)
#             self.saved_json_file["insertion_box"].append(insertion_box)
#             self.saved_json_file["insertion_cls"].append(insertion_cls)
            
#             deletion_boxer = bounding_boxes_reverse[arg_max_index].cpu().numpy()
#             deletion_box_id = (ious_reverse[arg_max_index] * cls_score_reverse[arg_max_index]).argmax().cpu().item()
#             deletion_box = deletion_boxer[deletion_box_id].astype(int).tolist()
#             deletion_iou = ious_reverse[arg_max_index][deletion_box_id].cpu().item()
#             deletion_cls = cls_score_reverse[arg_max_index][deletion_box_id].cpu().item()
#             self.saved_json_file["deletion_iou"].append(deletion_iou)
#             self.saved_json_file["deletion_box"].append(deletion_box)
#             self.saved_json_file["deletion_cls"].append(deletion_cls)
            
#             self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].cpu().item())
#             self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].cpu().item())
#             self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].cpu().item())
#             self.saved_json_file["region_area"].append(self.refer_baseline.sum() / self.region_area)
            
#             # Update S_set and baseline
#             S_set.append(self.V_set[arg_max_index])
#             self.refer_baseline = self.refer_baseline + self.V_set[arg_max_index]
#             del self.V_set[arg_max_index]

#         return S_set
    
#     def get_merge_set(self):
#         S_set = []
#         self.refer_baseline = np.zeros_like(self.V_set[0])
        
#         for i in tqdm(range(self.saved_json_file["sub-region_number"]), desc="Processing sub-regions"):
#             S_set = self.evaluation_maximun_sample(S_set)
        
#         self.saved_json_file["org_score"] = self.saved_json_file["insertion_score"][-1]
#         self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_score"][-1]
        
#         return S_set
    
#     def __call__(self, image, image_proccess, V_set, class_id, given_box):
#         """_summary_

#         Args:
#             image: cv2 format (H, W, 3)
#             image_proccess: torch.Tensor [3, H, W]
#             V_set: List of superpixel masks [n, H, W, 1]
#             class_id: List[int] (e.g., [43] for fork)
#             given_box: List[float] in xyxy format
#         """
#         self.save_file_init()
#         self.saved_json_file["target_box"] = given_box
#         self.saved_json_file["sub-region_number"] = len(V_set)
        
#         self.source_image = image
#         self.source_image_proccess = image_proccess
#         self.h, self.w, _ = self.source_image.shape
#         self.region_area = image_proccess.shape[1] * image_proccess.shape[2]
        
#         self.V_set = V_set.copy()
#         # Ensure target_label is a single integer
#         if isinstance(class_id, list):
#             if len(class_id) != 1:
#                 raise ValueError(f"Expected single class_id, got {class_id}")
#             self.target_label = class_id[0]
#         else:
#             self.target_label = class_id
#         self.target_box = torch.tensor(given_box, device=self.device)
#         self.saved_json_file["target_label"] = self.target_label
        
#         Submodular_Subset = self.get_merge_set()
        
#         self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
#         self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
#         return Submodular_Subset, self.saved_json_file

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

class DetectionSubModularExplanation:
    """
    Instance-level interpretability for object detection 
    based on submodular subset selection
    """
    def __init__(self, 
                 detection_model,
                 preproccessing_function,
                 lambda1=1.0,
                 lambda2=1.0,
                 batch_size=4,
                 mode="object",
                 device="cuda:3"):
        self.detection_model = detection_model.to(device)
        self.preproccessing_function = preproccessing_function
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.mode = mode
        
    def save_file_init(self):
        self.saved_json_file = {}
        self.saved_json_file["insertion_score"] = []
        self.saved_json_file["deletion_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["insertion_iou"] = []
        self.saved_json_file["insertion_box"] = []
        self.saved_json_file["insertion_cls"] = []
        self.saved_json_file["deletion_iou"] = []
        self.saved_json_file["deletion_box"] = []
        self.saved_json_file["deletion_cls"] = []
        self.saved_json_file["region_area"] = []
        self.saved_json_file["target_box"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["mode"] = self.mode
    
    def process_in_batches(self, images, batch_size, detection_model, h, w):
        all_bounding_boxes = []
        all_logits = []
        all_positive_maps = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_images = images[i * batch_size:(i + 1) * batch_size].to(self.device)
            print(f"Batch {i+1}/{num_batches} shape: {batch_images.shape}, device: {batch_images.device}")
            with torch.no_grad():
                try:
                    bounding_boxes, logits, positive_map = detection_model(batch_images, h, w)
                except Exception as e:
                    print(f"Error in detection_model for batch {i+1}: {e}")
                    bounding_boxes = torch.zeros((batch_images.shape[0], 1, 4), device=self.device)
                    logits = torch.zeros((batch_images.shape[0], 1, 256), device=self.device)
                    positive_map = None
            print(f"Batch {i+1} output: boxes shape: {bounding_boxes.shape}, logits shape: {logits.shape}, positive_map: {'None' if positive_map is None else positive_map.shape}")
            all_bounding_boxes.append(bounding_boxes)
            all_logits.append(logits)
            all_positive_maps.append(positive_map)

        all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0).to(self.device)
        all_logits = torch.cat(all_logits, dim=0).to(self.device)
        
        # Handle positive_map concatenation more carefully
        if all_positive_maps[0] is not None:
            try:
                all_positive_maps = torch.cat(all_positive_maps, dim=0).to(self.device)
            except:
                all_positive_maps = all_positive_maps[0]  # Use first batch's positive_map
        else:
            all_positive_maps = None
            
        print(f"Concatenated boxes shape: {all_bounding_boxes.shape}, logits shape: {all_logits.shape}, positive_map shape: {all_positive_maps.shape if all_positive_maps is not None else 'None'}")
        return all_bounding_boxes, all_logits, all_positive_maps
    
    def calculate_iou(self, batched_boxes, target_box):
        batched_boxes = batched_boxes.to(self.device)  # [batch, np, 4]
        target_box = self.target_box  # [4]
        
        if not (target_box[2] >= target_box[0] and target_box[3] >= target_box[1]):
            raise ValueError(f"Invalid target_box {target_box}: x2 < x1 or y2 < y1")
        if not torch.all(batched_boxes[..., 2] >= batched_boxes[..., 0]) or not torch.all(batched_boxes[..., 3] >= batched_boxes[..., 1]):
            print("Warning: Invalid batched_boxes detected, clamping coordinates")
            batched_boxes[..., [0, 2]] = torch.clamp(batched_boxes[..., [0, 2]], 0, self.w)
            batched_boxes[..., [1, 3]] = torch.clamp(batched_boxes[..., [1, 3]], 0, self.h)

        x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
        tx1, ty1, tx2, ty2 = target_box[0], target_box[1], target_box[2], target_box[3]

        inter_x1 = torch.maximum(x1, tx1)
        inter_y1 = torch.maximum(y1, ty1)
        inter_x2 = torch.minimum(x2, tx2)
        inter_y2 = torch.minimum(y2, ty2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        box_area = (x2 - x1) * (y2 - y1)
        target_area = (tx2 - tx1) * (ty2 - ty1)
        union_area = box_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        print(f"calculate_iou: batched_boxes shape: {batched_boxes.shape}, target_box: {target_box}, iou shape: {iou.shape}, max iou: {iou.max().item():.4f}")
        return iou
    
    def generate_masked_input(self, alpha_batch, source_image_process):
        alpha_array = np.stack(alpha_batch, axis=0)
        alpha_batch = torch.from_numpy(alpha_array).to(dtype=torch.float32, device=self.device)
        alpha_batch = alpha_batch.permute(0, 3, 1, 2)
        alpha_batch = alpha_batch.repeat(1, 3, 1, 1)
        source_image_process = source_image_process.unsqueeze(0).repeat(alpha_batch.size(0), 1, 1, 1).to(self.device)
        print(f"generate_masked_input: alpha_batch shape: {alpha_batch.shape}, device: {alpha_batch.device}")
        print(f"generate_masked_input: source_image_process shape: {source_image_process.shape}, device: {source_image_process.device}")
        return alpha_batch * source_image_process
    
    def safe_get_best_box_index(self, bounding_boxes, ious, cls_score, batch_idx):
        """Safely find the best bounding box index with proper bounds checking."""
        try:
            # Get boxes and scores for this batch
            batch_boxes = bounding_boxes[batch_idx]  # [num_proposals, 4]
            batch_ious = ious[batch_idx]            # [num_proposals]
            batch_cls = cls_score[batch_idx]        # [num_proposals]
            
            # Find valid boxes (IoU > threshold)
            valid_mask = batch_ious > 0.01
            
            if not valid_mask.any():
                print(f"Warning: No valid boxes found for batch {batch_idx}")
                return [0, 0, 0, 0], 0.0, 0.0
            
            # Get valid indices
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            valid_ious = batch_ious[valid_mask]
            valid_cls = batch_cls[valid_mask]
            
            # Calculate scores and find best
            score_product = valid_ious * valid_cls
            best_idx_in_valid = score_product.argmax().item()
            
            # Ensure we don't go out of bounds
            if best_idx_in_valid >= len(valid_indices):
                print(f"Warning: best_idx_in_valid {best_idx_in_valid} >= valid_indices length {len(valid_indices)}")
                best_idx_in_valid = len(valid_indices) - 1
            
            # Get the actual index in the original boxes
            best_idx = valid_indices[best_idx_in_valid].item()
            
            # Extract box coordinates and scores
            best_box = batch_boxes[best_idx].cpu().numpy().astype(float).tolist()
            best_iou = valid_ious[best_idx_in_valid].item()
            best_cls_score = valid_cls[best_idx_in_valid].item()
            
            return best_box, best_iou, best_cls_score
            
        except Exception as e:
            print(f"Error in safe_get_best_box_index for batch {batch_idx}: {e}")
            return [0, 0, 0, 0], 0.0, 0.0
    
    def evaluation_maximum_sample(self, S_set):
        print("Starting evaluation_maximum_sample")
        V_set_tem = [v for v in self.V_set]
        if not V_set_tem:
            print("Error: V_set is empty, cannot evaluate maximum sample.")
            return S_set, False
        print(f"V_set_tem length: {len(V_set_tem)}")

        alpha_batch = [(v + self.refer_baseline).astype(np.float32) for v in V_set_tem]
        
        batch_input_images = self.generate_masked_input(alpha_batch, self.source_image_proccess).to(self.device)
        batch_input_reverse = self.generate_masked_input([1 - a for a in alpha_batch], self.source_image_proccess).to(self.device)
        
        torch.cuda.empty_cache()
        
        # Insertion
        bounding_boxes, logits, positive_map = self.process_in_batches(batch_input_images, self.batch_size, self.detection_model, self.h, self.w)
        print(f"evaluation_maximum_sample: logits shape: {logits.shape}, target_label: {self.target_label}, positive_map shape: {positive_map.shape if positive_map is not None else 'None'}")
        print(f"logits max: {logits.max().item():.4f}, min: {logits.min().item():.4f}, shape: {logits.shape}")
        
        ious = self.calculate_iou(bounding_boxes, self.target_box)
        if self.mode == "cls":
            ious_clip = (ious > 0.5).float()
        elif self.mode == "object":
            ious_clip = ious
        print(f"ious_clip shape: {ious_clip.shape}, max: {ious_clip.max().item():.4f}")
        
        # Safely compute classification scores
        try:
            if positive_map is not None:
                # Ensure target_label is within bounds
                if self.target_label >= positive_map.shape[0]:
                    print(f"Warning: target_label={self.target_label} exceeds positive_map dimension ({positive_map.shape[0]})")
                    cls_score = torch.zeros(ious.shape, device=self.device)
                else:
                    token_indices = positive_map[self.target_label].nonzero(as_tuple=True)[0]
                    print(f"token_indices length: {len(token_indices)}, first 5: {token_indices[:5].tolist() if len(token_indices) > 0 else 'None'}")
                    if len(token_indices) == 0:
                        print(f"Warning: No token indices for target_label={self.target_label}, defaulting to zero scores")
                        cls_score = torch.zeros(ious.shape, device=self.device)
                    else:
                        # Ensure token indices are within logits bounds
                        valid_token_indices = token_indices[token_indices < logits.shape[2]]
                        if len(valid_token_indices) == 0:
                            cls_score = torch.zeros(ious.shape, device=self.device)
                        else:
                            cls_score = logits[:, :, valid_token_indices].max(dim=-1)[0].max(dim=-1)[0]
                            cls_score = cls_score.unsqueeze(1).expand(-1, ious.shape[1])
            else:
                print(f"positive_map is None, using max logits across all tokens for target_label={self.target_label}")
                if self.target_label >= logits.shape[2]:
                    print(f"Error: target_label={self.target_label} exceeds logits dimension ({logits.shape[2]})")
                    cls_score = torch.zeros(ious.shape, device=self.device)
                else:
                    cls_score = logits[:, :, self.target_label].max(dim=-1)[0]
                    cls_score = cls_score.unsqueeze(1).expand(-1, ious.shape[1])
            print(f"cls_score shape: {cls_score.shape}, max: {cls_score.max().item():.4f}")
        except Exception as e:
            print(f"Error in cls_score computation: target_label={self.target_label}, logits shape={logits.shape}: {e}")
            cls_score = torch.zeros(ious.shape, device=self.device)
        
        try:
            insertion_scores = (ious_clip * cls_score).max(dim=-1)[0]
            print(f"insertion_scores shape: {insertion_scores.shape}, max: {insertion_scores.max().item():.4f}")
        except Exception as e:
            print(f"Error in insertion_scores computation: {e}")
            insertion_scores = torch.zeros(ious.shape[0], device=self.device)
        
        # Deletion
        bounding_boxes_reverse, logits_reverse, _ = self.process_in_batches(batch_input_reverse, self.batch_size, self.detection_model, self.h, self.w)
        print(f"Deletion: logits_reverse shape: {logits_reverse.shape}, max: {logits_reverse.max().item():.4f}")
        ious_reverse = self.calculate_iou(bounding_boxes_reverse, self.target_box)
        if self.mode == "cls":
            ious_reverse_clip = (ious_reverse > 0.5).float()
        elif self.mode == "object":
            ious_reverse_clip = ious_reverse
        print(f"ious_reverse_clip shape: {ious_reverse_clip.shape}, max: {ious_reverse_clip.max().item():.4f}")
        
        # Compute deletion classification scores
        try:
            if positive_map is not None and self.target_label < positive_map.shape[0]:
                token_indices = positive_map[self.target_label].nonzero(as_tuple=True)[0]
                if len(token_indices) > 0:
                    valid_token_indices = token_indices[token_indices < logits_reverse.shape[2]]
                    if len(valid_token_indices) > 0:
                        cls_score_reverse = logits_reverse[:, :, valid_token_indices].max(dim=-1)[0].max(dim=-1)[0]
                        cls_score_reverse = cls_score_reverse.unsqueeze(1).expand(-1, ious_reverse.shape[1])
                    else:
                        cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
                else:
                    cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
            else:
                if self.target_label < logits_reverse.shape[2]:
                    cls_score_reverse = logits_reverse[:, :, self.target_label].max(dim=-1)[0]
                    cls_score_reverse = cls_score_reverse.unsqueeze(1).expand(-1, ious_reverse.shape[1])
                else:
                    cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
            print(f"cls_score_reverse shape: {cls_score_reverse.shape}, max: {cls_score_reverse.max().item():.4f}")
        except Exception as e:
            print(f"Error in cls_score_reverse computation: {e}")
            cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
        
        try:
            deletion_scores = (ious_reverse_clip * cls_score_reverse).max(dim=-1)[0]
            print(f"deletion_scores shape: {deletion_scores.shape}, max: {deletion_scores.max().item():.4f}")
        except Exception as e:
            print(f"Error in deletion_scores computation: {e}")
            deletion_scores = torch.zeros(ious_reverse.shape[0], device=self.device)
        
        # Overall submodular score
        try:
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
            print(f"smdl_scores shape: {smdl_scores.shape}, max: {smdl_scores.max().item():.4f}")
            
            if smdl_scores.numel() == 0:
                print("Error: smdl_scores is empty")
                return S_set, False
                
            arg_max_index = smdl_scores.argmax().item()
            print(f"arg_max_index: {arg_max_index}, V_set_tem length: {len(V_set_tem)}")
        except Exception as e:
            print(f"Error in smdl_scores computation: {e}")
            return S_set, False
        
        # Validate indices
        if arg_max_index >= len(V_set_tem):
            print(f"Error: arg_max_index {arg_max_index} exceeds V_set_tem length {len(V_set_tem)}")
            return S_set, False
        
        # Safe extraction of best boxes
        insertion_box, insertion_iou, insertion_cls = self.safe_get_best_box_index(
            bounding_boxes, ious, cls_score, arg_max_index
        )
        deletion_box, deletion_iou, deletion_cls = self.safe_get_best_box_index(
            bounding_boxes_reverse, ious_reverse, cls_score_reverse, arg_max_index
        )
        
        # Save results
        self.saved_json_file["insertion_iou"].append(insertion_iou)
        self.saved_json_file["insertion_box"].append(insertion_box)
        self.saved_json_file["insertion_cls"].append(insertion_cls)
        self.saved_json_file["deletion_iou"].append(deletion_iou)
        self.saved_json_file["deletion_box"].append(deletion_box)
        self.saved_json_file["deletion_cls"].append(deletion_cls)
        
        # Update S_set and V_set
        print(f"Updating S_set with V_set_tem[{arg_max_index}]")
        S_set.append(V_set_tem[arg_max_index])
        self.refer_baseline = self.refer_baseline + V_set_tem[arg_max_index]
        print(f"Removing V_set[{arg_max_index}], remaining length: {len(self.V_set)-1}")
        
        # Safe removal from V_set
        if arg_max_index < len(self.V_set):
            del self.V_set[arg_max_index]
        else:
            print(f"Warning: Cannot remove index {arg_max_index} from V_set of length {len(self.V_set)}")
        
        self.saved_json_file["region_area"].append(
            self.refer_baseline.sum() / self.region_area
        )
        
        # Safe indexing for scores
        if arg_max_index < len(insertion_scores):
            self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].item())
        else:
            self.saved_json_file["insertion_score"].append(0.0)
            
        if arg_max_index < len(deletion_scores):
            self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].item())
        else:
            self.saved_json_file["deletion_score"].append(0.0)
            
        if arg_max_index < len(smdl_scores):
            self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].item())
        else:
            self.saved_json_file["smdl_score"].append(0.0)
        
        print(f"Iteration: Insertion IoU: {insertion_iou:.4f}, Cls score: {insertion_cls:.4f}, Deletion IoU: {deletion_iou:.4f}, Deletion score: {deletion_cls:.4f}")
        
        return S_set, True
    
    def get_merge_set(self):
        S_set = []
        self.refer_baseline = np.zeros_like(self.V_set[0])
        
        max_iterations = len(self.V_set)
        for iteration in tqdm(range(max_iterations), desc="Processing sub-regions"):
            if not self.V_set:  # Check if V_set is empty
                print("V_set is empty, stopping iterations")
                break
                
            S_set, success = self.evaluation_maximum_sample(S_set)
            if not success:
                print("Breaking due to evaluation failure.")
                break
        
        # Safe access to scores
        if self.saved_json_file["insertion_score"]:
            self.saved_json_file["org_score"] = self.saved_json_file["insertion_score"][-1]
        else:
            self.saved_json_file["org_score"] = 0.0
            
        if self.saved_json_file["deletion_score"]:
            self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_score"][-1]
        else:
            self.saved_json_file["baseline_score"] = 0.0
        
        return S_set
    
    def __call__(self, image, image_proccess, V_set, class_id, given_box):
        # Input validation
        if not isinstance(given_box, (list, tuple)) or len(given_box) != 4:
            raise ValueError(f"Invalid given_box: {given_box}. Expected a list/tuple of 4 floats [x1, y1, x2, y2].")
        if not isinstance(class_id, int):
            raise ValueError(f"Invalid class_id: {class_id}. Expected a single integer.")
        if not V_set:
            raise ValueError("V_set is empty.")
        if not (given_box[2] >= given_box[0] and given_box[3] >= given_box[1]):
            raise ValueError(f"Invalid bounding box {given_box}: x2 < x1 or y2 < y1")
        
        print(f"Input image shape: {image.shape}, image_proccess shape: {image_proccess.shape}, device: {image_proccess.device}")
        self.save_file_init()
        self.saved_json_file["target_box"] = given_box
        self.saved_json_file["sub-region"] = len(V_set)
        
        self.source_image = image
        self.source_image_proccess = image_proccess.to(self.device)
        print(f"source_image_proccess moved to {self.source_image_proccess.device}")
        self.h, self.w = self.source_image.shape[:2]
        self.region_area = self.source_image_proccess.shape[1] * self.source_image_proccess.shape[2]
        
        self.V_set = [v for v in V_set]  # Create a copy to avoid modifying original
        self.target_label = torch.tensor(class_id, dtype=torch.long, device=self.device)
        self.target_box = torch.tensor(given_box, dtype=torch.float32, device=self.device)
        print(f"target_label shape: {self.target_label.shape}, device: {self.target_label.device}, value: {self.target_label}")
        print(f"target_box shape: {self.target_box.shape}, device: {self.target_box.device}, value: {self.target_box}")
        
        Submodular_Subset = self.get_merge_set()
        
        # Safe computation of max scores
        if self.saved_json_file["smdl_score"]:
            self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
            self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        else:
            self.saved_json_file["smdl_score_max"] = 0
            self.saved_json_file["smdl_score_max_index"] = -1
        
        print(f"Insertion IoU max: {max(self.saved_json_file['insertion_iou']) if self.saved_json_file['insertion_iou'] else 0:.4f}")
        print(f"Cls score max: {max(self.saved_json_file['insertion_cls']) if self.saved_json_file['insertion_cls'] else 0:.4f}")
        print(f"Deletion IoU max: {max(self.saved_json_file['deletion_iou']) if self.saved_json_file['deletion_iou'] else 0:.4f}")
        print(f"Cls score deletion max: {max(self.saved_json_file['deletion_cls']) if self.saved_json_file['deletion_cls'] else 0:.4f}")
        print(f"SMDL score max: {self.saved_json_file['smdl_score_max']:.4f}, Selected index: {self.saved_json_file['smdl_score_max_index']}")
        
        return Submodular_Subset, self.saved_json_file

# import torch
# import torch.nn.functional as F
# import numpy as np
# from tqdm import tqdm
# import cv2

# class DetectionSubModularExplanation:
#     """
#     Instance-level interpretability for object detection 
#     based on submodular subset selection
#     """
#     def __init__(self, 
#                  detection_model,
#                  preproccessing_function,
#                  lambda1=1.0,
#                  lambda2=1.0,
#                  lambda3=1.0,  # New parameter for confidence score
#                  batch_size=4,
#                  mode="object",
#                  device="cuda:0"):
#         self.detection_model = detection_model.to(device)
#         self.preproccessing_function = preproccessing_function
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.lambda3 = lambda3  # Weight for confidence score
#         self.device = torch.device(device)
#         self.batch_size = batch_size
#         self.mode = mode
        
#     def save_file_init(self):
#         self.saved_json_file = {}
#         self.saved_json_file["insertion_score"] = []
#         self.saved_json_file["deletion_score"] = []
#         self.saved_json_file["confidence_score"] = []  # New field for confidence score
#         self.saved_json_file["smdl_score"] = []
#         self.saved_json_file["insertion_iou"] = []
#         self.saved_json_file["insertion_box"] = []
#         self.saved_json_file["insertion_cls"] = []
#         self.saved_json_file["deletion_iou"] = []
#         self.saved_json_file["deletion_box"] = []
#         self.saved_json_file["deletion_cls"] = []
#         self.saved_json_file["region_area"] = []
#         self.saved_json_file["target_box"] = []
#         self.saved_json_file["lambda1"] = self.lambda1
#         self.saved_json_file["lambda2"] = self.lambda2
#         self.saved_json_file["lambda3"] = self.lambda3  # Save lambda3
#         self.saved_json_file["mode"] = self.mode
    

#     def compute_uncertainty(self, input_images, n_samples=3, scale=0.5):
#         """
#         Compute uncertainty by perturbing model weights and measuring gradient norms.
#         Args:
#             input_images (torch.Tensor): Shape (batch, 3, H, W)
#             n_samples (int): Number of perturbation samples
#             scale (float): Scale factor for weight noise
#         Returns:
#             numpy.ndarray: Uncertainty values for each input in the batch
#         """
#         batch_size = input_images.size(0)
#         uncertainties = np.zeros(batch_size)
#         valid_samples = 0
        
#         # Validate input tensor
#         if torch.isnan(input_images).any() or torch.isinf(input_images).any():
#             print(f"Input tensor contains NaN or Inf, shape: {input_images.shape}, aborting", flush=True)
#             return np.random.uniform(0.1, 0.5, batch_size)

#         # Save original model weights
#         original_state_dict = {k: v.clone() for k, v in self.detection_model.state_dict().items()}

#         def normalize_tensor(tensor, epsilon=1e-6):
#             tensor = tensor - tensor.min()
#             tensor = tensor / (tensor.max() + epsilon)
#             return tensor

#         for i in range(n_samples):
#             # Restore original weights
#             self.detection_model.load_state_dict(original_state_dict)

#             # Perturb weights
#             for name, param in self.detection_model.named_parameters():
#                 if param.requires_grad:
#                     weight_magnitude = param.abs().mean().clamp(min=1e-6)
#                     random_factor = torch.empty(1).uniform_(0.5, 1.0).item()
#                     noise_scale = scale * weight_magnitude * random_factor
#                     noise = torch.randn_like(param) * noise_scale
#                     new_param = param + noise
#                     clipping_range = 1.0 * param.abs().max().clamp(min=1e-6)
#                     new_param = torch.clamp(new_param, -clipping_range, clipping_range)
#                     param.data.copy_(new_param)

#             try:
#                 # Forward pass
#                 input_images.requires_grad_(True)
#                 bounding_boxes, logits, positive_map = self.detection_model(input_images, self.h, self.w)
#                 print(f"Sample {i}: boxes shape: {bounding_boxes.shape}, logits shape: {logits.shape}, positive_map: {positive_map}")
                
#                 # Compute classification score
#                 if positive_map is None:
#                     print(f"Sample {i}: positive_map is None, using max logits", flush=True)
#                     if logits.shape[1] == 0:  # No detections
#                         print(f"Sample {i}: No detections (logits shape: {logits.shape}), skipping", flush=True)
#                         continue
#                     cls_score = logits.max(dim=-1)[0].max(dim=-1)[0]  # Max across all proposals and tokens
#                 else:
#                     if self.target_label < positive_map.shape[0]:
#                         token_indices = positive_map[self.target_label].nonzero(as_tuple=True)[0]
#                         if len(token_indices) > 0 and token_indices.max() < logits.shape[2]:
#                             cls_score = logits[:, :, token_indices].max(dim=-1)[0].max(dim=-1)[0]
#                         else:
#                             print(f"Sample {i}: Invalid token indices for target_label={self.target_label}, using max logits", flush=True)
#                             cls_score = logits.max(dim=-1)[0].max(dim=-1)[0]
#                     else:
#                         print(f"Sample {i}: target_label={self.target_label} exceeds positive_map dim ({positive_map.shape[0]}), using max logits", flush=True)
#                         cls_score = logits.max(dim=-1)[0].max(dim=-1)[0]

#                 if torch.isnan(cls_score).any() or torch.isinf(cls_score).any():
#                     print(f"Sample {i}: cls_score contains NaN or Inf, skipping", flush=True)
#                     continue

#                 # Compute gradients
#                 loss = cls_score.mean()
#                 gradients = torch.autograd.grad(
#                     outputs=loss,
#                     inputs=input_images,
#                     create_graph=False,
#                     retain_graph=False,
#                     only_inputs=True,
#                     allow_unused=True
#                 )[0]

#                 if gradients is None:
#                     print(f"Sample {i}: Gradients are None, skipping", flush=True)
#                     continue

#                 if torch.isnan(gradients).any() or torch.isinf(gradients).any():
#                     print(f"Sample {i}: Gradients contain NaN or Inf, skipping", flush=True)
#                     continue

#                 gradient_norm = torch.sqrt((gradients ** 2).sum(dim=[1, 2, 3]))
#                 norm_gradient = normalize_tensor(gradient_norm, epsilon=1e-6)
#                 uncertainties += norm_gradient.detach().cpu().numpy()
#                 valid_samples += 1
#                 print(f"Sample {i}: Gradient norm min={norm_gradient.min().item():.4f}, max={norm_gradient.max().item():.4f}, cls_score max={cls_score.max().item():.4f}", flush=True)

#             except Exception as e:
#                 print(f"Sample {i}: Forward pass or gradient computation failed: {str(e)}", flush=True)
#                 continue
#             finally:
#                 input_images.requires_grad_(False)

#         self.detection_model.load_state_dict(original_state_dict)

#         if valid_samples > 0:
#             uncertainties /= valid_samples
#             uncertainties = np.clip(uncertainties, 0.0, 1.0)
#         else:
#             print("No valid samples produced, using random uncertainties", flush=True)
#             uncertainties = np.random.uniform(0.1, 0.5, batch_size)

#         if np.any(np.isnan(uncertainties)) or np.any(np.isinf(uncertainties)):
#             print("NaN or Inf in uncertainties, replacing with random values", flush=True)
#             uncertainties = np.random.uniform(0.1, 0.5, batch_size)

#         print(f"Final uncertainties: min={uncertainties.min():.4f}, max={uncertainties.max():.4f}, mean={uncertainties.mean():.4f}", flush=True)
#         return uncertainties
    
#     def process_in_batches(self, images, batch_size, detection_model, h, w):
#         # ... (unchanged, kept for reference)
#         all_bounding_boxes = []
#         all_logits = []
#         all_positive_maps = []
#         num_batches = (len(images) + batch_size - 1) // batch_size

#         for i in range(num_batches):
#             batch_images = images[i * batch_size:(i + 1) * batch_size].to(self.device)
#             print(f"Batch {i+1}/{num_batches} shape: {batch_images.shape}, device: {batch_images.device}")
#             with torch.no_grad():
#                 try:
#                     bounding_boxes, logits, positive_map = detection_model(batch_images, h, w)
#                 except Exception as e:
#                     print(f"Error in detection_model for batch {i+1}: {e}")
#                     bounding_boxes = torch.zeros((batch_images.shape[0], 1, 4), device=self.device)
#                     logits = torch.zeros((batch_images.shape[0], 1, 256), device=self.device)
#                     positive_map = None
#             print(f"Batch {i+1} output: boxes shape: {bounding_boxes.shape}, logits shape: {logits.shape}, positive_map: {'None' if positive_map is None else positive_map.shape}")
#             all_bounding_boxes.append(bounding_boxes)
#             all_logits.append(logits)
#             all_positive_maps.append(positive_map)

#         all_bounding_boxes = torch.cat(all_bounding_boxes, dim=0).to(self.device)
#         all_logits = torch.cat(all_logits, dim=0).to(self.device)
        
#         if all_positive_maps[0] is not None:
#             try:
#                 all_positive_maps = torch.cat(all_positive_maps, dim=0).to(self.device)
#             except:
#                 all_positive_maps = all_positive_maps[0]
#         else:
#             all_positive_maps = None
            
#         print(f"Concatenated boxes shape: {all_bounding_boxes.shape}, logits shape: {all_logits.shape}, positive_map shape: {all_positive_maps.shape if all_positive_maps is not None else 'None'}")
#         return all_bounding_boxes, all_logits, all_positive_maps
    
#     def calculate_iou(self, batched_boxes, target_box):
#         # ... (unchanged)
#         batched_boxes = batched_boxes.to(self.device)
#         target_box = self.target_box
        
#         if not (target_box[2] >= target_box[0] and target_box[3] >= target_box[1]):
#             raise ValueError(f"Invalid target_box {target_box}: x2 < x1 or y2 < y1")
#         if not torch.all(batched_boxes[..., 2] >= batched_boxes[..., 0]) or not torch.all(batched_boxes[..., 3] >= batched_boxes[..., 1]):
#             print("Warning: Invalid batched_boxes detected, clamping coordinates")
#             batched_boxes[..., [0, 2]] = torch.clamp(batched_boxes[..., [0, 2]], 0, self.w)
#             batched_boxes[..., [1, 3]] = torch.clamp(batched_boxes[..., [1, 3]], 0, self.h)

#         x1, y1, x2, y2 = batched_boxes[..., 0], batched_boxes[..., 1], batched_boxes[..., 2], batched_boxes[..., 3]
#         tx1, ty1, tx2, ty2 = target_box[0], target_box[1], target_box[2], target_box[3]

#         inter_x1 = torch.maximum(x1, tx1)
#         inter_y1 = torch.maximum(y1, ty1)
#         inter_x2 = torch.minimum(x2, tx2)
#         inter_y2 = torch.minimum(y2, ty2)

#         inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
#         box_area = (x2 - x1) * (y2 - y1)
#         target_area = (tx2 - tx1) * (ty2 - ty1)
#         union_area = box_area + target_area - inter_area

#         iou = inter_area / (union_area + 1e-6)
#         print(f"calculate_iou: batched_boxes shape: {batched_boxes.shape}, target_box: {target_box}, iou shape: {iou.shape}, max iou: {iou.max().item():.4f}")
#         return iou
    
#     def generate_masked_input(self, alpha_batch, source_image_process):
#         # ... (unchanged)
#         alpha_array = np.stack(alpha_batch, axis=0)
#         alpha_batch = torch.from_numpy(alpha_array).to(dtype=torch.float32, device=self.device)
#         alpha_batch = alpha_batch.permute(0, 3, 1, 2)
#         alpha_batch = alpha_batch.repeat(1, 3, 1, 1)
#         source_image_process = source_image_process.unsqueeze(0).repeat(alpha_batch.size(0), 1, 1, 1).to(self.device)
#         print(f"generate_masked_input: alpha_batch shape: {alpha_batch.shape}, device: {alpha_batch.device}")
#         print(f"generate_masked_input: source_image_process shape: {source_image_process.shape}, device: {source_image_process.device}")
#         return alpha_batch * source_image_process
    
#     def safe_get_best_box_index(self, bounding_boxes, ious, cls_score, batch_idx):
#         # ... (unchanged)
#         try:
#             batch_boxes = bounding_boxes[batch_idx]
#             batch_ious = ious[batch_idx]
#             batch_cls = cls_score[batch_idx]
            
#             valid_mask = batch_ious > 0.01
            
#             if not valid_mask.any():
#                 print(f"Warning: No valid boxes found for batch {batch_idx}")
#                 return [0, 0, 0, 0], 0.0, 0.0
            
#             valid_indices = valid_mask.nonzero(as_tuple=True)[0]
#             valid_ious = batch_ious[valid_mask]
#             valid_cls = batch_cls[valid_mask]
            
#             score_product = valid_ious * valid_cls
#             best_idx_in_valid = score_product.argmax().item()
            
#             if best_idx_in_valid >= len(valid_indices):
#                 print(f"Warning: best_idx_in_valid {best_idx_in_valid} >= valid_indices length {len(valid_indices)}")
#                 best_idx_in_valid = len(valid_indices) - 1
            
#             best_idx = valid_indices[best_idx_in_valid].item()
            
#             best_box = batch_boxes[best_idx].cpu().numpy().astype(float).tolist()
#             best_iou = valid_ious[best_idx_in_valid].item()
#             best_cls_score = valid_cls[best_idx_in_valid].item()
            
#             return best_box, best_iou, best_cls_score
            
#         except Exception as e:
#             print(f"Error in safe_get_best_box_index for batch {batch_idx}: {e}")
#             return [0, 0, 0, 0], 0.0, 0.0
    
#     def evaluation_maximum_sample(self, S_set):
#         print("Starting evaluation_maximum_sample")
#         V_set_tem = [v for v in self.V_set]
#         if not V_set_tem:
#             print("Error: V_set is empty, cannot evaluate maximum sample.")
#             return S_set, False
#         print(f"V_set_tem length: {len(V_set_tem)}")

#         alpha_batch = [(v + self.refer_baseline).astype(np.float32) for v in V_set_tem]
        
#         batch_input_images = self.generate_masked_input(alpha_batch, self.source_image_proccess).to(self.device)
#         batch_input_reverse = self.generate_masked_input([1 - a for a in alpha_batch], self.source_image_proccess).to(self.device)
        
#         # Compute uncertainty
#         confidence_scores = 1 - self.compute_uncertainty(batch_input_images)
#         print(f"confidence_scores shape: {confidence_scores.shape}, max: {np.max(confidence_scores):.4f}")
        
#         torch.cuda.empty_cache()
        
#         # Insertion
#         bounding_boxes, logits, positive_map = self.process_in_batches(batch_input_images, self.batch_size, self.detection_model, self.h, self.w)
#         print(f"evaluation_maximum_sample: logits shape: {logits.shape}, target_label: {self.target_label}, positive_map shape: {positive_map.shape if positive_map is not None else 'None'}")
#         print(f"logits max: {logits.max().item():.4f}, min: {logits.min().item():.4f}, shape: {logits.shape}")
        
#         ious = self.calculate_iou(bounding_boxes, self.target_box)
#         if self.mode == "cls":
#             ious_clip = (ious > 0.5).float()
#         elif self.mode == "object":
#             ious_clip = ious
#         print(f"ious_clip shape: {ious_clip.shape}, max: {ious_clip.max().item():.4f}")
        
#         try:
#             if positive_map is not None:
#                 if self.target_label >= positive_map.shape[0]:
#                     print(f"Warning: target_label={self.target_label} exceeds positive_map dimension ({positive_map.shape[0]})")
#                     cls_score = torch.zeros(ious.shape, device=self.device)
#                 else:
#                     token_indices = positive_map[self.target_label].nonzero(as_tuple=True)[0]
#                     print(f"token_indices length: {len(token_indices)}, first 5: {token_indices[:5].tolist() if len(token_indices) > 0 else 'None'}")
#                     if len(token_indices) == 0:
#                         print(f"Warning: No token indices for target_label={self.target_label}, defaulting to zero scores")
#                         cls_score = torch.zeros(ious.shape, device=self.device)
#                     else:
#                         valid_token_indices = token_indices[token_indices < logits.shape[2]]
#                         if len(valid_token_indices) == 0:
#                             cls_score = torch.zeros(ious.shape, device=self.device)
#                         else:
#                             cls_score = logits[:, :, valid_token_indices].max(dim=-1)[0].max(dim=-1)[0]
#                             cls_score = cls_score.unsqueeze(1).expand(-1, ious.shape[1])
#             else:
#                 print(f"positive_map is None, using max logits across all tokens for target_label={self.target_label}")
#                 if self.target_label >= logits.shape[2]:
#                     print(f"Error: target_label={self.target_label} exceeds logits dimension ({logits.shape[2]})")
#                     cls_score = torch.zeros(ious.shape, device=self.device)
#                 else:
#                     cls_score = logits[:, :, self.target_label].max(dim=-1)[0]
#                     cls_score = cls_score.unsqueeze(1).expand(-1, ious.shape[1])
#             print(f"cls_score shape: {cls_score.shape}, max: {cls_score.max().item():.4f}")
#         except Exception as e:
#             print(f"Error in cls_score computation: target_label={self.target_label}, logits shape={logits.shape}: {e}")
#             cls_score = torch.zeros(ious.shape, device=self.device)
        
#         try:
#             insertion_scores = (ious_clip * cls_score).max(dim=-1)[0]
#             print(f"insertion_scores shape: {insertion_scores.shape}, max: {insertion_scores.max().item():.4f}")
#         except Exception as e:
#             print(f"Error in insertion_scores computation: {e}")
#             insertion_scores = torch.zeros(ious.shape[0], device=self.device)
        
#         # Deletion
#         bounding_boxes_reverse, logits_reverse, _ = self.process_in_batches(batch_input_reverse, self.batch_size, self.detection_model, self.h, self.w)
#         print(f"Deletion: logits_reverse shape: {logits_reverse.shape}, max: {logits_reverse.max().item():.4f}")
#         ious_reverse = self.calculate_iou(bounding_boxes_reverse, self.target_box)
#         if self.mode == "cls":
#             ious_reverse_clip = (ious_reverse > 0.5).float()
#         elif self.mode == "object":
#             ious_reverse_clip = ious_reverse
#         print(f"ious_reverse_clip shape: {ious_reverse_clip.shape}, max: {ious_reverse_clip.max().item():.4f}")
        
#         try:
#             if positive_map is not None and self.target_label < positive_map.shape[0]:
#                 token_indices = positive_map[self.target_label].nonzero(as_tuple=True)[0]
#                 if len(token_indices) > 0:
#                     valid_token_indices = token_indices[token_indices < logits_reverse.shape[2]]
#                     if len(valid_token_indices) > 0:
#                         cls_score_reverse = logits_reverse[:, :, valid_token_indices].max(dim=-1)[0].max(dim=-1)[0]
#                         cls_score_reverse = cls_score_reverse.unsqueeze(1).expand(-1, ious_reverse.shape[1])
#                     else:
#                         cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
#                 else:
#                     cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
#             else:
#                 if self.target_label < logits_reverse.shape[2]:
#                     cls_score_reverse = logits_reverse[:, :, self.target_label].max(dim=-1)[0]
#                     cls_score_reverse = cls_score_reverse.unsqueeze(1).expand(-1, ious_reverse.shape[1])
#                 else:
#                     cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
#             print(f"cls_score_reverse shape: {cls_score_reverse.shape}, max: {cls_score_reverse.max().item():.4f}")
#         except Exception as e:
#             print(f"Error in cls_score_reverse computation: {e}")
#             cls_score_reverse = torch.zeros(ious_reverse.shape, device=self.device)
        
#         try:
#             deletion_scores = (ious_reverse_clip * cls_score_reverse).max(dim=-1)[0]
#             print(f"deletion_scores shape: {deletion_scores.shape}, max: {deletion_scores.max().item():.4f}")
#         except Exception as e:
#             print(f"Error in deletion_scores computation: {e}")
#             deletion_scores = torch.zeros(ious_reverse.shape[0], device=self.device)
        
#         # Overall submodular score with confidence term
#         try:
#             confidence_scores_tensor = torch.from_numpy(confidence_scores).to(self.device)
#             smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores) + self.lambda3 * confidence_scores_tensor
#             print(f"smdl_scores shape: {smdl_scores.shape}, max: {smdl_scores.max().item():.4f}")
            
#             if smdl_scores.numel() == 0:
#                 print("Error: smdl_scores is empty")
#                 return S_set, False
                
#             arg_max_index = smdl_scores.argmax().item()
#             print(f"arg_max_index: {arg_max_index}, V_set_tem length: {len(V_set_tem)}")
#         except Exception as e:
#             print(f"Error in smdl_scores computation: {e}")
#             return S_set, False
        
#         # Safe extraction of best boxes
#         insertion_box, insertion_iou, insertion_cls = self.safe_get_best_box_index(
#             bounding_boxes, ious, cls_score, arg_max_index
#         )
#         deletion_box, deletion_iou, deletion_cls = self.safe_get_best_box_index(
#             bounding_boxes_reverse, ious_reverse, cls_score_reverse, arg_max_index
#         )
        
#         # Save results
#         self.saved_json_file["insertion_iou"].append(insertion_iou)
#         self.saved_json_file["insertion_box"].append(insertion_box)
#         self.saved_json_file["insertion_cls"].append(insertion_cls)
#         self.saved_json_file["deletion_iou"].append(deletion_iou)
#         self.saved_json_file["deletion_box"].append(deletion_box)
#         self.saved_json_file["deletion_cls"].append(deletion_cls)
#         self.saved_json_file["confidence_score"].append(float(confidence_scores[arg_max_index]))  # Save confidence score
        
#         # Update S_set and V_set
#         print(f"Updating S_set with V_set_tem[{arg_max_index}]")
#         S_set.append(V_set_tem[arg_max_index])
#         self.refer_baseline = self.refer_baseline + V_set_tem[arg_max_index]
#         print(f"Removing V_set[{arg_max_index}], remaining length: {len(self.V_set)-1}")
        
#         if arg_max_index < len(self.V_set):
#             del self.V_set[arg_max_index]
#         else:
#             print(f"Warning: Cannot remove index {arg_max_index} from V_set of length {len(self.V_set)}")
        
#         self.saved_json_file["region_area"].append(
#             self.refer_baseline.sum() / self.region_area
#         )
        
#         if arg_max_index < len(insertion_scores):
#             self.saved_json_file["insertion_score"].append(insertion_scores[arg_max_index].item())
#         else:
#             self.saved_json_file["insertion_score"].append(0.0)
            
#         if arg_max_index < len(deletion_scores):
#             self.saved_json_file["deletion_score"].append(deletion_scores[arg_max_index].item())
#         else:
#             self.saved_json_file["deletion_score"].append(0.0)
            
#         if arg_max_index < len(smdl_scores):
#             self.saved_json_file["smdl_score"].append(smdl_scores[arg_max_index].item())
#         else:
#             self.saved_json_file["smdl_score"].append(0.0)
        
#         print(f"Iteration: Insertion IoU: {insertion_iou:.4f}, Cls score: {insertion_cls:.4f}, Deletion IoU: {deletion_iou:.4f}, Deletion score: {deletion_cls:.4f}, Confidence score: {confidence_scores[arg_max_index]:.4f}")
        
#         return S_set, True
    
#     def get_merge_set(self):
#         # ... (unchanged)
#         S_set = []
#         self.refer_baseline = np.zeros_like(self.V_set[0])
        
#         max_iterations = len(self.V_set)
#         for iteration in tqdm(range(max_iterations), desc="Processing sub-regions"):
#             if not self.V_set:
#                 print("V_set is empty, stopping iterations")
#                 break
                
#             S_set, success = self.evaluation_maximum_sample(S_set)
#             if not success:
#                 print("Breaking due to evaluation failure.")
#                 break
        
#         if self.saved_json_file["insertion_score"]:
#             self.saved_json_file["org_score"] = self.saved_json_file["insertion_score"][-1]
#         else:
#             self.saved_json_file["org_score"] = 0.0
            
#         if self.saved_json_file["deletion_score"]:
#             self.saved_json_file["baseline_score"] = self.saved_json_file["deletion_score"][-1]
#         else:
#             self.saved_json_file["baseline_score"] = 0.0
        
#         return S_set
    
#     def __call__(self, image, image_proccess, V_set, class_id, given_box):
#         # ... (unchanged)
#         if not isinstance(given_box, (list, tuple)) or len(given_box) != 4:
#             raise ValueError(f"Invalid given_box: {given_box}. Expected a list/tuple of 4 floats [x1, y1, x2, y2].")
#         if not isinstance(class_id, int):
#             raise ValueError(f"Invalid class_id: {class_id}. Expected a single integer.")
#         if not V_set:
#             raise ValueError("V_set is empty.")
#         if not (given_box[2] >= given_box[0] and given_box[3] >= given_box[1]):
#             raise ValueError(f"Invalid bounding box {given_box}: x2 < x1 or y2 < y1")
        
#         print(f"Input image shape: {image.shape}, image_proccess shape: {image_proccess.shape}, device: {image_proccess.device}")
#         self.save_file_init()
#         self.saved_json_file["target_box"] = given_box
#         self.saved_json_file["sub-region"] = len(V_set)
        
#         self.source_image = image
#         self.source_image_proccess = image_proccess.to(self.device)
#         print(f"source_image_proccess moved to {self.source_image_proccess.device}")
#         self.h, self.w = self.source_image.shape[:2]
#         self.region_area = self.source_image_proccess.shape[1] * self.source_image_proccess.shape[2]
        
#         self.V_set = [v for v in V_set]
#         self.target_label = torch.tensor(class_id, dtype=torch.long, device=self.device)
#         self.target_box = torch.tensor(given_box, dtype=torch.float32, device=self.device)
#         print(f"target_label shape: {self.target_label.shape}, device: {self.target_label.device}, value: {self.target_label}")
#         print(f"target_box shape: {self.target_box.shape}, device: {self.target_box.device}, value: {self.target_box}")
        
#         Submodular_Subset = self.get_merge_set()
        
#         if self.saved_json_file["smdl_score"]:
#             self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
#             self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
#         else:
#             self.saved_json_file["smdl_score_max"] = 0
#             self.saved_json_file["smdl_score_max_index"] = -1
        
#         print(f"Insertion IoU max: {max(self.saved_json_file['insertion_iou']) if self.saved_json_file['insertion_iou'] else 0:.4f}")
#         print(f"Cls score max: {max(self.saved_json_file['insertion_cls']) if self.saved_json_file['insertion_cls'] else 0:.4f}")
#         print(f"Deletion IoU max: {max(self.saved_json_file['deletion_iou']) if self.saved_json_file['deletion_iou'] else 0:.4f}")
#         print(f"Cls score deletion max: {max(self.saved_json_file['deletion_cls']) if self.saved_json_file['deletion_cls'] else 0:.4f}")
#         print(f"SMDL score max: {self.saved_json_file['smdl_score_max']:.4f}, Selected index: {self.saved_json_file['smdl_score_max_index']}")
        
#         return Submodular_Subset, self.saved_json_file