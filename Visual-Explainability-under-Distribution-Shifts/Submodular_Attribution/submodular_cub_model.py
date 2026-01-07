# import json
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import functorch as AF
# import math
# import random
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision import models
# import torch.nn as nn

# import tensorflow as tf
# from tensorflow.keras.applications import ResNet101
# from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from collections import OrderedDict

# import tensorflow_addons as tfa
# from keras.models import load_model
# from insight_face_models import *

# import time

# import torchvision.transforms as transforms
# from .evidential import relu_evidence, exp_evidence

# from tqdm import tqdm

# class HibCriterion(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, z_samples, alpha, beta, indices_tuple):

#         n_samples = z_samples.shape[1]
                
#         if len(indices_tuple) == 3:
#             a, p, n = indices_tuple
#             ap = an = a
#         elif len(indices_tuple) == 4:
#             ap, p, an, n = indices_tuple
        
#         alpha = torch.nn.functional.softplus(alpha)

#         loss = 0
#         for i in range(n_samples):
#             z_i = z_samples[:, i, :]
#             for j in range(n_samples):
#                 z_j = z_samples[:, j, :]
        
#                 prob_pos = torch.sigmoid(- alpha * torch.sum((z_i[ap] - z_j[p])**2, dim=1) + beta) + 1e-6
#                 prob_neg = torch.sigmoid(- alpha * torch.sum((z_i[an] - z_j[n])**2, dim=1) + beta) + 1e-6
                
#                 # maximize the probability of positive pairs and minimize the probability of negative pairs
#                 loss += -torch.log(prob_pos) - torch.log(1 - prob_neg)
#         loss = loss / (n_samples ** 2)
        
#         return loss.mean()

# class PatchInfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z_samples):
#         # Compute cosine similarity matrix
#         sim_matrix = F.cosine_similarity(z_samples.unsqueeze(1), z_samples.unsqueeze(0), dim=-1)

#         # Positive mask (diagonal entries are positive pairs)
#         pos_mask = torch.eye(z_samples.shape[0], device=z_samples.device)

#         # Compute InfoNCE loss
#         exp_sim = torch.exp(sim_matrix / self.temperature)
#         loss = -torch.log(exp_sim / (exp_sim.sum(dim=1) - exp_sim.diag()))
#         return loss.mean()
    


# class LaplaceApproximation:
#     def __init__(self, model, hessian_approximation='positive', prior_precision=1.0):
#         """
#         Initialize the Laplace Approximation for uncertainty estimation
        
#         Args:
#             model: PyTorch model to approximate
#             hessian_approximation: Which Hessian approximation to use ('positive', 'fixed', or 'full')
#             prior_precision: Precision of the prior distribution (1/variance)
#         """
#         self.model = model
#         self.hessian_approximation = hessian_approximation
#         self.prior_precision = prior_precision
#         self.device = next(model.parameters()).device
        
#         # Store MAP weights and initialize precision (diagonal of Hessian)
#         self.mean_weights = {}
#         self.precision = {}
        
#         # Store the current weights as MAP estimate
#         for name, param in self.model.named_parameters():
#             self.mean_weights[name] = param.data.clone()
#             self.precision[name] = torch.ones_like(param.data) * prior_precision
    
#     def update_precision(self, loss, data_loader):
#         """
#         Update the precision parameters (Hessian) using the selected approximation
#         """
#         # Reset precision to prior
#         for name in self.precision:
#             self.precision[name] = torch.ones_like(self.precision[name]) * self.prior_precision
            
#         # Accumulate precision over batches
#         for batch in data_loader:
#             inputs, indices_tuple = batch[0].to(self.device), batch[1]
            
#             # Get embeddings for input batch
#             embeddings = self.model(inputs)
            
#             # Compute Hessian based on selected approximation
#             if self.hessian_approximation == 'positive':
#                 # Only use positive pairs for Hessian
#                 if len(indices_tuple) == 3:
#                     a, p, _ = indices_tuple
#                     ap = a
#                 elif len(indices_tuple) == 4:
#                     ap, p, _, _ = indices_tuple
                    
#                 # Compute positive pair gradients and accumulate Hessian
#                 for i in range(len(ap)):
#                     self._accumulate_hessian(embeddings[ap[i]], embeddings[p[i]], positive=True)
                    
#             elif self.hessian_approximation == 'fixed':
#                 # Handle both positive and negative pairs with fixed cross derivatives
#                 if len(indices_tuple) == 3:
#                     a, p, n = indices_tuple
#                     ap = an = a
#                 elif len(indices_tuple) == 4:
#                     ap, p, an, n = indices_tuple
                    
#                 # Compute positive pair gradients
#                 for i in range(len(ap)):
#                     self._accumulate_hessian(embeddings[ap[i]], embeddings[p[i]], positive=True, fixed=True)
                
#                 # Compute negative pair gradients
#                 for i in range(len(an)):
#                     # Only consider negatives within margin
#                     distance = torch.sum((embeddings[an[i]] - embeddings[n[i]])**2)
#                     if distance < 1.0:  # Margin
#                         self._accumulate_hessian(embeddings[an[i]], embeddings[n[i]], positive=False, fixed=True)
            
#             elif self.hessian_approximation == 'full':
#                 # Use full Hessian with ReLU to ensure positive definiteness
#                 if len(indices_tuple) == 3:
#                     a, p, n = indices_tuple
#                     ap = an = a
#                 elif len(indices_tuple) == 4:
#                     ap, p, an, n = indices_tuple
                    
#                 # Compute positive pair gradients
#                 for i in range(len(ap)):
#                     self._accumulate_hessian(embeddings[ap[i]], embeddings[p[i]], positive=True)
                
#                 # Compute negative pair gradients
#                 for i in range(len(an)):
#                     # Only consider negatives within margin
#                     distance = torch.sum((embeddings[an[i]] - embeddings[n[i]])**2)
#                     if distance < 1.0:  # Margin
#                         self._accumulate_hessian(embeddings[an[i]], embeddings[n[i]], positive=False)
    
#     def _accumulate_hessian(self, z_i, z_j, positive=True, fixed=False):
#         """
#         Accumulate Hessian contribution from a pair of embeddings
#         """
#         # Create computation graph for pair
#         pair_distance = torch.sum((z_i - z_j)**2)
        
#         # Compute gradients
#         self.model.zero_grad()
#         pair_distance.backward(retain_graph=True)
        
#         # Get gradients and accumulate diagonal Hessian
#         for name, param in self.model.named_parameters():
#             if param.grad is not None:
#                 grad = param.grad.clone()
                
#                 if fixed:
#                     # Fixed approximation: Only consider diagonal elements
#                     hessian_contrib = grad**2
#                 else:
#                     # Full or Positive approximation: Consider cross derivatives
#                     hessian_contrib = grad**2
                
#                 # Adjust sign based on positive/negative pair
#                 if not positive:
#                     hessian_contrib = -hessian_contrib
                
#                 # Ensure positive definiteness for full approximation
#                 if self.hessian_approximation == 'full' and not positive:
#                     hessian_contrib = torch.relu(hessian_contrib)
                
#                 # Accumulate precision
#                 self.precision[name] += hessian_contrib
    
#     def sample_weights(self, n_samples=10):
#         """
#         Sample weights from the Laplace approximation
#         """
#         samples = []
        
#         for i in range(n_samples):
#             # Store original weights
#             original_weights = {}
#             for name, param in self.model.named_parameters():
#                 original_weights[name] = param.data.clone()
            
#             # Sample new weights
#             for name, param in self.model.named_parameters():
#                 # Compute standard deviation for each weight
#                 std = 1.0 / torch.sqrt(self.precision[name] + 1e-8)
                
#                 # Sample from normal distribution
#                 noise = torch.randn_like(param.data) * std
                
#                 # Set sampled weights
#                 param.data = self.mean_weights[name] + noise
            
#             # Add sample index
#             samples.append(i)
            
#             # Restore original weights
#             for name, param in self.model.named_parameters():
#                 param.data = original_weights[name]
        
#         return samples

# class CubSubModularExplanationV2(object):
#     def __init__(self, 
#                  cfg_path="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
#                  k = 50,
#                  lambda1 = 1.0,
#                  lambda2 = 1.0,
#                  lambda3 = 1.0,
#                  lambda4 = 1.0):
#         super(CubSubModularExplanationV2, self).__init__()
        
#         # Load model configuration / 导入模型的配置文件
#         with open(cfg_path, "r", encoding="utf-8") as f:
#             self.cfg = json.load(f)

#         assert self.cfg["version"] == 2
        
#         self.device = torch.device(self.cfg["device"])
#         self.moda = self.cfg["mode"]

#         self.transforms = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#         ])
            
#         # Uncertainty estimation model / 不确定性估计模型
#         self.uncertainty_model = self.define_uncertrainty_network(
#             self.cfg["uncertainty_model"]["model_path"])
        
#         self.laplace_approx = LaplaceApproximation(
#             self.uncertainty_model, 
#             hessian_approximation='fixed',  # Use fixed Hessian approximation
#             prior_precision=1.0
#         )
#         # Face recognition
#         self.recognition_model = self.define_recognition_model(
#             self.cfg["recognition_model"]["num_classes"], self.cfg["recognition_model"]["model_path"])

#         # Parameters of the submodular / submodular的超参数
#         self.k = k
        
#         # Parameter of the LtLG algorithm / LtLG贪婪算法的参数
#         self.ltl_log_ep = 5
        
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.lambda3 = lambda3
#         self.lambda4 = lambda4

#         self.softmax = tf.keras.layers.Softmax(axis=-1)

#         if "resnet" in self.cfg["recognition_model"]["model_path"]:
#             from keras.applications.resnet import preprocess_input
#             self.preprocess_input = preprocess_input
#             self.tf_size = 224
#         elif "vgg19" in self.cfg["recognition_model"]["model_path"]:
#             from keras.applications.vgg19 import preprocess_input
#             self.preprocess_input = preprocess_input
#             self.tf_size = 224
#         elif "efficientnetv2" in self.cfg["recognition_model"]["model_path"]:
#             from keras.applications.efficientnet_v2 import preprocess_input
#             self.preprocess_input = preprocess_input
#             self.tf_size = 384
#         elif "mobilenetv2" in self.cfg["recognition_model"]["model_path"]:
#             from keras.applications.mobilenet_v2 import preprocess_input
#             self.preprocess_input = preprocess_input
#             self.tf_size = 224

#     def convert_prepare_image(self, image, size=224):
#         img = cv2.resize(image[...,::-1], (self.tf_size, self.tf_size))
#         img = self.preprocess_input(np.array(img))
        
#         return img
    
#     def preprocess_image_uncertainty(self, image, size=224):
#         img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         img = self.transforms(img).numpy()
#         return img
    
#     # def define_recognition_model(self, num_classes, pretrained_path):
#     #     """
#     #     init the face recognition model
#     #     """
#     #     model_base = load_model(pretrained_path)
#     #     layer_name = "dense"
#     #     # layer_name = "flatten"
#     #     model = tf.keras.models.Model(inputs=model_base.input, outputs=[model_base.get_layer(layer_name).output, model_base.output])
#     #     # model.layers[-1].activation = tf.keras.activations.linear
#     #     print("Success load pre-trained bird recognition model {}".format(pretrained_path))

#     #     return model
#     def define_recognition_model(self, num_classes, pretrained_path):
#         """
#         init the face recognition model
#         """
#         input_tensor = Input(shape=(224, 224, 3), name='input_1')
#         base_model = ResNet101(weights='imagenet', include_top=False, 
#                             input_tensor=input_tensor)
#         # base_model = ResNet101(weights='imagenet', include_top=False, 
#         #                       input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        
#         # Add custom classification head
#         x = base_model.output
#         x = GlobalAveragePooling2D()(x)
#         x = Dense(1024, activation='relu')(x)
#         x = Dense(512, activation='relu')(x)
#         class_number = 196
#         predictions = Dense(class_number, activation='softmax')(x)
        
#         # Create the final model
#         model_base = Model(inputs=base_model.input, outputs=predictions)
#         model_base.load_weights(pretrained_path)
#         # model_base = load_model(pretrained_path)
#         layer_name = "dense"
#         # layer_name = "flatten"
#         model = tf.keras.models.Model(inputs=model_base.input, outputs=[model_base.get_layer(layer_name).output, model_base.output])
#         # model.layers[-1].activation = tf.keras.activations.linear
#         print("Success load pre-trained bird recognition model {}".format(pretrained_path))

#         return model

#     # def define_uncertrainty_network(self, pretrained_path):
#     #     """
#     #     Init the uncertainty model
#     #     """
#     #     from torchvision import models
#     #     import torch.nn as nn
#     #     model=models.resnet101(pretrained=False) # resnet152
#     #     channel_in = model.fc.in_features
#     #     model.fc = nn.Linear(channel_in, self.cfg["uncertainty_model"]["num_classes"])

#     #     if pretrained_path is not None and os.path.exists(pretrained_path):
#     #         model_dict = model.state_dict()
#     #         pretrained_param = torch.load(pretrained_path, map_location=torch.device('cpu'))

#     #         try:
#     #             pretrained_param = pretrained_param.state_dict()
#     #         except:
#     #             pass
                
#     #         new_state_dict = OrderedDict()
#     #         for k, v in pretrained_param.items():
#     #             if k in model_dict:
#     #                 new_state_dict[k] = v
#     #                 # print("Load parameter {}".format(k))
#     #             elif k[7:] in model_dict:
#     #                 new_state_dict[k[7:]] = v
#     #                 # print("Load parameter {}".format(k[7:]))
#     #             else:
#     #                 print("Parameter {} has not been load".format(k))
#     #         model_dict.update(new_state_dict)
#     #         model.load_state_dict(model_dict)
#     #         print("Success load pre-trained uncertainty model {}".format(pretrained_path))
#     #     else:
#     #         print("not load pretrained")
        
#     #     model.eval()
#     #     model.to(self.device)

#     #     return model
    
#     def define_uncertrainty_network(self, pretrained_path):
#         """
#         Initialize the uncertainty model (ResNet101) with specified number of classes.
        
#         Args:
#             pretrained_path (str): Path to pre-trained model weights
            
#         Returns:
#             torch.nn.Module: Initialized ResNet101 model
#         """
#         # Initialize ResNet101
#         model = models.resnet101(pretrained=False)
#         channel_in = model.fc.in_features
        
#         # Set the fully connected layer to output num_classes (196 for Cars196)
#         num_classes = self.cfg["uncertainty_model"]["num_classes"]
#         model.fc = nn.Linear(channel_in, num_classes)

#         # Load pre-trained weights if available
#         if pretrained_path is not None and os.path.exists(pretrained_path):
#             try:
#                 pretrained_param = torch.load(pretrained_path, map_location=torch.device('cpu'))
                
#                 # Handle state_dict if saved as a model or state_dict
#                 if hasattr(pretrained_param, 'state_dict'):
#                     pretrained_param = pretrained_param.state_dict()
                    
#                 # Prepare new state dict, handling fc layer mismatch
#                 model_dict = model.state_dict()
#                 new_state_dict = OrderedDict()
#                 for k, v in pretrained_param.items():
#                     # Handle keys with or without 'module.' prefix (e.g., DataParallel)
#                     key = k if k in model_dict else k[7:] if k[7:] in model_dict else None
#                     if key is None:
#                         print(f"Parameter {k} not found in model, skipping")
#                         continue
                    
#                     # Check for fc layer mismatch
#                     if key == 'fc.weight' or key == 'fc.bias':
#                         if v.size() != model_dict[key].size():
#                             print(f"Skipping {key} due to size mismatch: "
#                                 f"pretrained {v.size()} vs model {model_dict[key].size()}")
#                             continue
                    
#                     new_state_dict[key] = v
                
#                 # Update model with compatible weights
#                 model_dict.update(new_state_dict)
#                 model.load_state_dict(model_dict, strict=False)  # Allow missing fc weights
#                 print(f"Successfully loaded pre-trained uncertainty model from {pretrained_path}")
                
#             except Exception as e:
#                 print(f"Error loading pre-trained weights from {pretrained_path}: {str(e)}")
#                 print("Proceeding with randomly initialized fc layer")
#         else:
#             print(f"No pre-trained weights found at {pretrained_path}, using random initialization")

#         # Set model to evaluation mode and move to device
#         model.eval()
#         model.to(self.device)

#         return model
    
#     def exp_evidence(self, y):
#         # 使用np.clip限制y的值在-10到10之间，然后计算指数
#         return np.exp(np.clip(y, -10, 10))
    
#     # def exp_evidence(self, y):
#     #     """Apply exponential to outputs to get evidence with clipping for numerical stability"""
#     #     if isinstance(y, torch.Tensor):
#     #         # Handle PyTorch tensor case - keep on the same device
#     #         return torch.exp(torch.clamp(y, -10.0, 10.0))
#     #     elif isinstance(y, np.ndarray):
#     #         # Handle NumPy array case
#     #         return np.exp(np.clip(y, -10.0, 10.0))
#     #     else:
#     #         # Try to convert to tensor first, sticking with original device
#     #         try:
#     #             y_tensor = torch.as_tensor(y, device=self.device)
#     #             return torch.exp(torch.clamp(y_tensor, -10.0, 10.0))
#     #         except:
#     #             # If all else fails, try to convert to CPU first, then to numpy
#     #             try:
#     #                 if hasattr(y, 'cpu'):
#     #                     y_cpu = y.cpu()
#     #                     return np.exp(np.clip(y_cpu.detach().numpy(), -10.0, 10.0))
#     #                 else:
#     #                     return np.exp(np.clip(np.array(y), -10.0, 10.0))
#     #             except Exception as e:
#     #                 print(f"Error in exp_evidence: {e}")
#     #                 # Return a safe default
#     #                 if hasattr(y, 'shape'):
#     #                     return torch.ones(y.shape, device=self.device)
#     #                 else:
#     #                     return 1.0

#     def compute_uncertainty(self, input_images, scale = 5):
#         """
#         Compute the uncertainty of the model
#         input: torch.Size(batch, 3, w, h)
#         """
#         # print(input_images)
#         # print(input_images.min(), input_images.max())
#         with torch.no_grad():
#             input_images = torch.from_numpy(input_images)
#             output = self.uncertainty_model(input_images.to(self.device))
#         evidence = exp_evidence(output)
#         alpha = evidence + 1
#         uncertainty = self.cfg["uncertainty_model"]["num_classes"] / torch.sum(alpha, dim=1, keepdim=True)

#         return uncertainty.reshape(-1).cpu().numpy()
    
#     # def compute_uncertainty(self, input_images, n_samples=10, scale=5):
#     #     """
#     #     Compute uncertainty using Laplace approximation for model weights
        
#     #     Args:
#     #         input_images: Input images in numpy array format (batch, 3, w, h)
#     #         n_samples: Number of weight samples to draw
#     #         scale: Scaling factor for uncertainty
            
#     #     Returns:
#     #         Uncertainty scores for each input image
#     #     """
#     #     # Convert numpy array to tensor
#     #     input_tensor = torch.from_numpy(input_images).to(self.device)
        
#     #     # Initialize container for multiple predictions
#     #     predictions = []
        
#     #     try:
#     #         # Get weight samples from Laplace approximation
#     #         with torch.no_grad():
#     #             # Restore original weights first to ensure clean state
#     #             for name, param in self.uncertainty_model.named_parameters():
#     #                 if name in self.laplace_approx.mean_weights:
#     #                     param.data.copy_(self.laplace_approx.mean_weights[name])
                
#     #             # First, get mean prediction with original weights
#     #             mean_output = self.uncertainty_model(input_tensor)
                
#     #             # Sample weights and make predictions
#     #             for _ in range(n_samples):
#     #                 # Store original weights
#     #                 original_weights = {}
#     #                 for name, param in self.uncertainty_model.named_parameters():
#     #                     original_weights[name] = param.data.clone()
                    
#     #                 # Sample new weights
#     #                 for name, param in self.uncertainty_model.named_parameters():
#     #                     # Compute standard deviation for each weight with numerical stability
#     #                     precision = torch.clamp(self.laplace_approx.precision[name], min=1e-6)
#     #                     std = 1.0 / torch.sqrt(precision)
                        
#     #                     # Cap extreme values
#     #                     std = torch.clamp(std, max=0.1)
                        
#     #                     # Sample from normal distribution with small noise magnitude
#     #                     noise = torch.randn_like(param.data).to(self.device) * std * 0.1
#     #                     param.data = self.laplace_approx.mean_weights[name] + noise
                    
#     #                 # Forward pass with current weights
#     #                 output = self.uncertainty_model(input_tensor)
#     #                 predictions.append(output)
                    
#     #                 # Restore original weights
#     #                 for name, param in self.uncertainty_model.named_parameters():
#     #                     param.data.copy_(original_weights[name])
                
#     #             # Stack predictions and compute statistics
#     #             predictions = torch.stack(predictions)  # (n_samples, batch_size, num_classes)
                
#     #             # Predictive variance (model uncertainty) - sum across classes
#     #             model_variance = torch.var(predictions, dim=0).sum(dim=1, keepdim=True)
                
#     #             # Evidence-based uncertainty (aleatoric uncertainty)
#     #             evidence = self.exp_evidence(mean_output)
#     #             alpha = evidence + 1.0
#     #             alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
#     #             dirichlet_uncertainty = self.cfg["uncertainty_model"]["num_classes"] / alpha_sum
                
#     #             # Combined uncertainty (weighted sum)
#     #             combined_uncertainty = 0.7 * dirichlet_uncertainty + 0.3 * scale * model_variance
                
#     #             # Ensure values are in reasonable range
#     #             combined_uncertainty = torch.clamp(combined_uncertainty, min=0.01, max=0.99)
                
#     #     except Exception as e:
#     #         print(f"Error in uncertainty computation: {str(e)}")
#     #         print("Falling back to simple uncertainty estimation")
            
#     #         # Fallback to simple uncertainty estimation
#     #         with torch.no_grad():
#     #             output = self.uncertainty_model(input_tensor)
#     #             evidence = self.exp_evidence(output)
#     #             alpha = evidence + 1.0
#     #             alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
#     #             combined_uncertainty = self.cfg["uncertainty_model"]["num_classes"] / alpha_sum
        
#     #     # Return as numpy array
#     #     return combined_uncertainty.reshape(-1).cpu().numpy()
    
    
#     # def compute_uncertainty(self, input_images, n_samples=10, scale=1, temperature=0.1, batch_size=8):
#     #     """
#     #     Compute uncertainty using MC Dropout, Laplace Approximation (LAM), and Probabilistic Feature Embeddings (PFE).
    
#     #     Args:
#     #         input_images (np.ndarray): Input images (batch, C, H, W)
#     #         n_samples (int): Number of MC Dropout stochastic passes
#     #         scale (float): Scaling factor for entropy-based regularization
#     #         temperature (float): Temperature scaling for prediction sharpness
#     #         batch_size (int): Batch size for Hessian computation to save memory
    
#     #     Returns:
#     #         dict: Dictionary containing combined uncertainty values.
#     #     """
    
#     #     input_images = torch.from_numpy(input_images).to(self.device)
    
#     #     # Enable MC Dropout
#     #     self.uncertainty_model.train()
    
#     #     # Perform multiple stochastic forward passes
#     #     outputs = []
#     #     for _ in range(n_samples):
#     #         with torch.no_grad():
#     #             output = self.uncertainty_model(input_images)
#     #             output /= temperature  # Apply temperature scaling
#     #             outputs.append(output)
    
#     #     # Compute Mean & Variance across MC Dropout samples
#     #     outputs = torch.stack(outputs)
#     #     mean_output = outputs.mean(dim=0)
#     #     var_output = outputs.var(dim=0)
    
#     #     # Compute Laplace Approximation Uncertainty (Hessian-based)
#     #     laplace_uncertainty = 1/(1+self.compute_hessian_diag(self.uncertainty_model, input_images, batch_size=batch_size, hessian_layers=2))
    
#     #     # Compute Probabilistic Feature Embedding (PFE) uncertainty
#     #     pfe_uncertainty = self.compute_pfe_uncertainty(input_images)
    
#     #     # Compute evidence and alpha for aleatoric uncertainty
#     #     evidence = torch.exp(torch.clamp(mean_output, -10, 10))
#     #     alpha = evidence + 1
#     #     num_classes = mean_output.size(1)
    
#     #     # Compute Aleatoric Uncertainty (data uncertainty)
#     #     aleatoric_uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    
#     #     # Compute Entropy Regularization
#     #     probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
#     #     entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)
    
#     #     # Compute Total Uncertainty
#     #     total_uncertainty = (
#     #         (var_output.mean(dim=1).cpu().numpy()
#     #         + aleatoric_uncertainty.squeeze().cpu().numpy()
#     #         + scale * entropy.squeeze().cpu().numpy()
#     #         + laplace_uncertainty
#     #         + pfe_uncertainty)/20
#     #     )
#     #     return total_uncertainty
    
#     # def compute_uncertainty(self, input_images, n_samples=10, scale=1, temperature=0.1, batch_size=4, mode="diagonal"):


#     #     input_images = torch.from_numpy(input_images).to(self.device)
    
#     #     # Enable MC Dropout
#     #     self.uncertainty_model.train()
    
#     #     # Perform multiple stochastic forward passes
#     #     outputs = []
#     #     for _ in range(n_samples):
#     #         with torch.no_grad():
#     #             output = self.uncertainty_model(input_images)
#     #             output /= temperature  # Apply temperature scaling
#     #             outputs.append(output)
    
#     #     # Compute Mean & Variance across MC Dropout samples
#     #     outputs = torch.stack(outputs)
#     #     mean_output = outputs.mean(dim=0)
#     #     var_output = outputs.var(dim=0)
    
#     #     # ✅ Compute Laplace Approximation (GGN for Contrastive Loss)
#     #     # hessian = self.compute_hessian_low_rank(self.uncertainty_model, input_images, rank =2, batch_size=batch_size, hessian_layers=3)
#     #     # hessian_diag = self.compute_hessian_diag(
#     #     #     self.uncertainty_model, input_images, batch_size=batch_size, hessian_layers=3
#     #     # )
#     #     hessian_diag = self.compute_hessian_contrastive(
#     #         self.uncertainty_model, input_images, batch_size=batch_size, hessian_layers=3, mode = "diagonal"
#     #     )
#     #     diag_hessian = hessian_diag.detach().cpu().numpy()
#     #     # diag_hessian_total += np.sum(diag_hessian)
#     #     # print(diag_hessian)
#     #     laplace_uncertainty = 1.0 / (1.0 + abs(np.mean(diag_hessian)))
#     #     # print(laplace_uncertainty)
#     #     # Compute Probabilistic Feature Embedding (PFE) uncertainty
#     #     pfe_uncertainty = self.compute_pfe_uncertainty(input_images)

#     #     # Compute evidence and alpha for aleatoric uncertainty
#     #     evidence = torch.exp(torch.clamp(mean_output, -10, 10))
#     #     alpha = evidence + 1
#     #     num_classes = mean_output.size(1)
    
#     #     # Compute Aleatoric Uncertainty (data uncertainty)
#     #     aleatoric_uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    
#     #     # Compute Entropy Regularization
#     #     probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
#     #     entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)

#     #     total_uncertainty = (
#     #         (aleatoric_uncertainty.squeeze().cpu().numpy()
#     #         + scale * entropy.squeeze().cpu().numpy()
#     #         + laplace_uncertainty  # ✅ Now properly scaled
#     #         + pfe_uncertainty)
#     #     )
#     #     return total_uncertainty

    
#     # def compute_hessian_diag_contrastive(self, model, inputs, batch_size=8, hessian_layers=2, mode="full"):
#     #     """
#     #     Compute the diagonal approximation of the Hessian for contrastive loss using GGN approximation.
    
#     #     Args:
#     #         model (torch.nn.Module): The neural network model.
#     #         inputs (torch.Tensor): Input batch.
#     #         batch_size (int): Number of images per mini-batch to reduce memory load.
#     #         hessian_layers (int): Number of layers to compute Hessian for (reduces memory usage).
#     #         mode (str): "positive", "fixed", or "full" - determines the Hessian approximation method.
    
#     #     Returns:
#     #         float: Trace of the diagonal Hessian (Laplace Approximation Uncertainty).
#     #     """
    
#     #     model.zero_grad()  # Clear previous gradients
#     #     inputs.requires_grad_(True)
    
#     #     # Select only the last 'hessian_layers' layers for Hessian computation
#     #     model_params = list(model.parameters())[-hessian_layers:]
    
#     #     diag_hessian_mean = 0
#     #     # print(len(inputs))
#     #     for i in range(0, len(inputs), batch_size):
#     #         batch = inputs[i : i + batch_size]
#     #         actual_batch_size = batch.shape[0] 
#     #         # print(batch.dim())
#     #         # print(self.target_label)
#     #         # ✅ Ensure `target_label` is expanded to match batch size
#     #         batch_labels = torch.full((actual_batch_size,), self.target_label, dtype=torch.long, device=batch.device)
#     #         # print(batch_labels)
#     #         # print("Batch shape before passing to model:", batch.shape)

#     #         # ✅ Forward pass through the model in `no_grad` mode to save memory
#     #         with torch.no_grad():
#     #             embeddings = model(batch)  # Extract feature embeddings
#     #             embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization
                
#     #         model.eval()

#     #         # ✅ Compute Jacobian for each image separately to reduce memory usage
#     #         def model_forward(x):
#     #             with torch.no_grad():
#     #                 return model(x)
    
#     #         # J_theta = torch.func.vmap(torch.func.jacrev(model_forward))(batch)
#     #         # print("J_theta shape after jacrev:", J_theta.shape)  # Check if it remains 4D
#     #         # J_theta = torch.func.vmap(lambda x: torch.func.jacrev(model_forward)(x))(batch)
            
#     #         # J_theta = torch.func.vmap(torch.func.jacrev(model_forward, argnums=None))(batch)
#     #         J_dash = torch.func.jacrev(model_forward)(batch)  # Compute for a single image
#     #         J_dash = J_dash.sum(dim=(1, 2))  # Sum over extra dimensions
#     #         # print("Updated J_single shape:", J_single.shape)# Should still be 4D
#     #         J_theta = torch.func.vmap(lambda x: x)(J_dash)
#     #         # print("J_theta shape after vmap:", J_theta.shape)  # Check final shape
#     #         J_theta = J_theta.view(J_theta.size(0), -1)  # Reshape Jacobian
    
#     #         # ✅ Identify positive and negative pairs using `batch_labels`
#     #         pos_mask = batch_labels.unsqueeze(1) == batch_labels.unsqueeze(0)  # Same class = Positive pair
#     #         neg_mask = ~pos_mask  # Different class = Negative pair
    
#     #         # ✅ Compute Hessian for positive and negative pairs
#     #         with torch.no_grad():  # Avoid storing unnecessary gradients
#     #             Hp_diag = torch.einsum("bi,bi->bi", J_theta, J_theta)  # Only diagonal elements
#     #             Hn_diag = -Hp_diag  # Negative pair approximation

#     #             # ✅ Construct diagonal Hessian matrices
#     #             # Hp = torch.diag_embed(Hp_diag)  # (batch_size, flattened_dim, flattened_dim)
#     #             # Hn = torch.diag_embed(Hn_diag)

#     #             # ✅ Apply Hessian approximation method
#     #             if mode == "positive":
#     #                 Hn_diag = torch.zeros_like(Hn_diag)  # Ignore negative pairs
#     #             elif mode == "full":
#     #                 Hp_diag = F.relu(Hp_diag)  # Ensure positive semidefiniteness
#     #                 Hn_diag = F.relu(Hn_diag)

#     #             # ✅ Compute diagonal Hessian trace
#     #             # hessian_diag = Hp * pos_mask + Hn * neg_mask
#     #             # Reshape masks to match dimensions
#     #             flattened_dim = Hp_diag.shape[1]  # Get the second dimension (feature size)

#     #             pos_mask = pos_mask.unsqueeze(2).expand(actual_batch_size, actual_batch_size, flattened_dim)  # ✅ Correct
#     #             neg_mask = neg_mask.unsqueeze(2).expand(actual_batch_size, actual_batch_size, flattened_dim)  # ✅ Correct

#     #             hessian_diag_trace = (Hp_diag * pos_mask + Hn_diag * neg_mask).sum(dim=1)

#     #             # ✅ Accumulate Hessian diagonal trace
#     #             diag_hessian_mean += hessian_diag_trace.mean().item()
    
#     #         # ✅ Free memory immediately after use
#     #         J_theta = J_theta.detach()
#     #         Hp_diag = Hp_diag.detach()
#     #         Hn_diag = Hn_diag.detach()
#     #         hessian_diag_trace = hessian_diag_trace.detach()
#     #         del J_theta, Hp_diag, Hn_diag, hessian_diag_trace

    
#     #     return diag_hessian_mean # Return trace of diagonal Hessian
    


#     # def compute_hessian_diag_contrastive(self, model, inputs, batch_size=8, hessian_layers=2, mode="fixed"):
#     #     """
#     #     Compute the diagonal approximation of the Hessian for contrastive loss using GGN approximation with HibCriterion.

#     #     Args:
#     #         model (torch.nn.Module): The neural network model.
#     #         inputs (torch.Tensor): Input batch.
#     #         batch_size (int): Number of images per mini-batch to reduce memory load.
#     #         hessian_layers (int): Number of layers to compute Hessian for (reduces memory usage).
#     #         mode (str): "positive", "fixed", or "full" - determines the Hessian approximation method.

#     #     Returns:
#     #         float: Trace of the diagonal Hessian (Laplace Approximation Uncertainty).
#     #     """
        
#     #     model.zero_grad()  # Clear previous gradients
#     #     inputs.requires_grad_(True)
        
#     #     # Select only the last 'hessian_layers' layers for Hessian computation
#     #     model_params = list(model.parameters())[-hessian_layers:]
#     #     # print(model_params)

#     #     # Initialize the contrastive loss function
#     #     contrastive_loss = HibCriterion()

#     #     diag_hessian_mean = 0

#     #     for i in range(0, len(inputs) - 1, batch_size):
#     #         batch = inputs[i : i + batch_size]

#     #         # Ensure `target_label` is expanded to match batch size
#     #         batch_labels = torch.full((batch.shape[0],), self.target_label, dtype=torch.long, device=batch.device)

#     #         # Forward pass through the model
#     #         # with torch.no_grad():
#     #         z_samples = model(batch)  # Extract feature embeddings
#     #         z_samples = F.normalize(z_samples, p=2, dim=1)  # L2 normalization
            
#     #         z_samples = z_samples.unsqueeze(1)  # Expanding to (batch_size, 1, feature_dim)

#     #         # Compute loss for contrastive learning
#     #         alpha = torch.tensor(1.0, requires_grad=True, device=batch.device)
#     #         beta = torch.tensor(0.5, requires_grad=True, device=batch.device)

#     #         # Create indices for positive and negative pairs
#     #         indices_tuple = (torch.arange(batch.shape[0], device=batch.device), 
#     #                         torch.arange(batch.shape[0], device=batch.device),
#     #                         torch.randint(0, batch.shape[0], (batch.shape[0],), device=batch.device))
            
#     #         for param in model_params:
#     #             param.requires_grad_(True)
#     #         loss = contrastive_loss(z_samples, alpha, beta, indices_tuple)
#     #         # print("Loss value:", loss.item())  # Check if it's nonzero


#     #         # Compute Jacobian of the loss function w.r.t model parameters

                
#     #         # print(loss.requires_grad)
#     #         J_loss = torch.autograd.grad(loss, model_params, create_graph=True)
            
#     #         # for param in model_params:
                
#     #         #     if param.grad is None:
#     #         #         print(f"Gradient is None for parameter {param}")
#     #         # Compute Hessian Diagonal Approximation
#     #         diag_hessian = []
#     #         for grad in J_loss:
#     #             # print(grad)
#     #             if grad is not None:
#     #                 # print("Madhav")
#     #                 hess_diag = torch.autograd.grad(grad, model_params, grad_outputs=torch.ones_like(grad), retain_graph=True, allow_unused=True)
#     #                 diag_hessian.append(
#     #                     torch.cat([h.flatten() if h is not None else torch.zeros_like(g.flatten()) for h, g in zip(hess_diag, J_loss)])
#     #                 )
#     #             else:
#     #                 # print("MG")
#     #                 diag_hessian.append(torch.zeros_like(model_params[0]))
            
#     #         # Convert Hessian diagonal to numpy and sum up
#     #         diag_hessian = torch.cat(diag_hessian).detach().cpu().numpy()
#     #         diag_hessian_mean += np.mean(diag_hessian)

#     #         # Free GPU memory
#     #         del J_loss, diag_hessian
#     #         torch.cuda.empty_cache()

#     #     return diag_hessian_mean  # Return trace of diagonal Hessian
    

#     # def compute_hessian_contrastive(self, model, inputs, batch_size=4, hessian_layers=2, mode="diagonal"):
        
#     #     model.zero_grad()
#     #     inputs.requires_grad_(True)

#     #     # Select last `hessian_layers` layers
#     #     model_params = list(model.parameters())[-hessian_layers:]

#     #     # Use PatchInfoNCELoss
#     #     contrastive_loss = PatchInfoNCELoss()

#     #     # Hessian storage
#     #     if mode == "diagonal":
#     #         hessian_diag = torch.zeros(sum(p.numel() for p in model_params), device=inputs.device)
#     #     else:
#     #         hessian_full_sum = torch.zeros((sum(p.numel() for p in model_params), 
#     #                                     sum(p.numel() for p in model_params)), device=inputs.device)

#     #     for i in range(0, len(inputs) - 1, batch_size):
#     #         batch = inputs[i : i + batch_size]
#     #         # print(batch.min(), batch.max())
#     #         z_samples = model(batch)  # Extract embeddings
#     #         z_samples = F.normalize(z_samples, p=2, dim=1)  # L2 normalization

#     #         # Compute loss
#     #         loss = contrastive_loss(z_samples)

#     #         # Hessian calculation
#     #         if mode == "diagonal":
#     #             grads = torch.autograd.grad(loss, model_params, create_graph=True)
#     #             idx = 0
#     #             for grad, param in zip(grads, model_params):
#     #                 hess_diag_block = torch.autograd.grad(grad, param, grad_outputs=torch.ones_like(grad), retain_graph=True)[0]
#     #                 numel = param.numel()
#     #                 hessian_diag[idx : idx + numel] = hess_diag_block.view(-1)
#     #                 idx += numel
#     #         else:
#     #             hessian_full = hessian(lambda params: contrastive_loss(z_samples), tuple(model_params))
#     #             hessian_flat = torch.cat([h.view(-1) for row in hessian_full for h in row], dim=0).reshape(hessian_full_sum.shape)
#     #             hessian_full_sum += hessian_flat.detach()

#     #         # Memory cleanup
#     #         torch.cuda.empty_cache()

#     #     return hessian_diag if mode == "diagonal" else hessian_full_sum




    
#     def compute_hessian_contrastive(self, model, inputs, batch_size=4, hessian_layers=2, mode = "diagonal"):
#         """
#         Compute the diagonal approximation of the Hessian matrix using mini-batches to reduce memory usage.
    
#         Args:
#             model (torch.nn.Module): The neural network model.
#             inputs (torch.Tensor): Input batch.
#             batch_size (int): Number of images per mini-batch to reduce memory load.
#             hessian_layers (int): Number of layers to compute Hessian for (reduces memory usage).
    
#         Returns:
#             float: Trace of the diagonal Hessian (Laplace Approximation Uncertainty).
#         """
    
#         model.zero_grad()  # Clear previous gradients
    
#         # Ensure inputs require gradients
#         inputs.requires_grad_(True)
    
#         # Select only the last 'hessian_layers' layers for Hessian computation
#         model_params = list(model.parameters())[-hessian_layers:]
    
#         # Process Hessian in small mini-batches
#         diag_hessian_total = 0
#         for i in range(0, len(inputs), batch_size):
#             batch = inputs[i : i + batch_size]
    
#             # Forward pass
#             output = model(batch)
#             # print(output)
#             # print(torch.argmax(output, dim=1))
#             loss = F.cross_entropy(output, torch.argmax(output, dim=1))  # Compute loss
    
#             # Compute first derivative (gradients w.r.t. selected model parameters)
#             grads = torch.autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    
#             # Compute second derivative (diagonal Hessian approximation)
#             diag_hessian = []
#             for grad in grads:
#                 hess_diag = torch.autograd.grad(
#                     grad,
#                     model_params,
#                     grad_outputs=torch.ones_like(grad),
#                     retain_graph=True,  # Do NOT retain graph to save memory
#                     allow_unused=True
#                 )
#                 diag_hessian.append(
#                     torch.cat([
#                         h.flatten() if h is not None else torch.zeros_like(g.flatten()) 
#                         for h, g in zip(hess_diag, grads)
#                     ])
#                 )
    
#             # Convert Hessian diagonal to numpy and sum up
#             diag_hessian = torch.cat(diag_hessian)
#             # diag_hessian = torch.cat(diag_hessian).detach().cpu().numpy()
#             # diag_hessian_total += np.sum(diag_hessian)
    
#             # # Free GPU memory
#             # del grads, hess_diag, diag_hessian
#             # torch.cuda.empty_cache()
    
#         return diag_hessian  # Return trace of diagonal Hessian
    
#     def compute_hessian_low_rank(self, model, inputs, rank=2, batch_size=4, hessian_layers=3):
        
#         model.zero_grad()
#         inputs.requires_grad_(True)
#         model_params = list(model.parameters())[-hessian_layers:]

#         # Sample random vectors for low-rank approximation
#         V = torch.randn(sum(p.numel() for p in model_params), rank, device=inputs.device)
#         HV = torch.zeros_like(V)

#         for i in range(0, len(inputs), batch_size):
#             batch = inputs[i : i + batch_size]
#             output = model(batch)
#             loss = F.cross_entropy(output, torch.argmax(output, dim=1))

#             grads = torch.autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
#             grads_flat = torch.cat([g.flatten() for g in grads])

#             for j in range(rank):
#                 hvp = torch.autograd.grad(grads_flat, model_params, grad_outputs=V[:, j], retain_graph=True)
#                 hvp_flat = torch.cat([h.flatten() if h is not None else torch.zeros_like(g.flatten()) for h, g in zip(hvp, grads)])
#                 HV[:, j] += hvp_flat

#         # Low-rank approximation: H ≈ HV @ V^T
#         # HV = HV.cpu()
#         # V = V.cpu()
#         # HV = HV.to(torch.bfloat16)
#         # V = V.to(torch.bfloat16)

#         # low_rank_hessian = torch.matmul(HV, V.T)
#         HV_q = torch.quantize_per_tensor(HV, scale=0.1, zero_point=0, dtype=torch.qint8)
#         V_q = torch.quantize_per_tensor(V, scale=0.1, zero_point=0, dtype=torch.qint8)

#         low_rank_hessian_q = torch.matmul(HV_q.dequantize(), V_q.dequantize().T)
#         # low_rank_hessian = HV @ V.T
#         # low_rank_hessian = torch.zeros((HV.shape[0], HV.shape[0]), device=HV.device)
#         # for i in range(0, HV.shape[0], 1000):  # Process 1000 rows at a time
#         #     low_rank_hessian[i : i+1000, :] = HV[i : i+1000] @ V.T

#         return low_rank_hessian_q


    
#     def compute_pfe_uncertainty(self, input_images):
#         """
#         Compute Probabilistic Feature Embedding (PFE) Uncertainty.

#         Args:
#             input_images (torch.Tensor): Input images.

#         Returns:
#             float: PFE-based uncertainty.
#         """
#         with torch.no_grad():
#             features = self.uncertainty_model(input_images)  # Extract features
#             # print(features.shape)
#             feature_var = features.var(dim=0).mean().cpu().numpy()  # Compute variance across feature space
#         return feature_var

#     # def compute_pfe_uncertainty(self, input_images):
#     #     """
#     #     Compute Probabilistic Feature Embedding (PFE) Uncertainty.

#     #     Args:
#     #         input_images (torch.Tensor): Input images.

#     #     Returns:
#     #         float: PFE-based uncertainty.
#     #     """
#     #     with torch.no_grad():
#     #         features = self.uncertainty_model(input_images)  # Extract feature embeddings
            
#     #         mu = features.mean(dim=0)  # Compute mean feature vector
#     #         sigma2 = F.softplus(features.var(dim=0))  # Compute variance, ensure positivity
        
#     #     # Define sample pairs (indices) for computing uncertainty
#     #     batch_size = features.shape[0]
#     #     if batch_size % 2 == 1:
#     #         batch_size -= 1  # Make it even by removing last sample
#     #     indices_tuple = (torch.arange(batch_size // 2), torch.arange(batch_size // 2, batch_size))

#     #     # Compute PFE uncertainty using the loss formula
#     #     pfe_loss = (mu[indices_tuple[0]] - mu[indices_tuple[1]])**2 / (sigma2[indices_tuple[0]] + sigma2[indices_tuple[1]])
#     #     pfe_loss += torch.log(sigma2[indices_tuple[0]] + sigma2[indices_tuple[1]])

#     #     return pfe_loss.mean().cpu().numpy()  # Return PFE uncertainty

    
    
#     def compute_effectiveness_score(self, features):
#         """
#         Computes Eeffectiveness Score: The point should be distant from all the other elements in the subset.
#         features: torch.Size(batch, d)
#         """
#         if self.cfg["effectiveness_distance_metric"] == "cosine":
#             norm_feature = tf.nn.l2_normalize(features, axis=1)
#             # Consine Similarity
#             cosine_similarity = tf.matmul(norm_feature, tf.transpose(norm_feature))
#             cosine_similarity = tf.clip_by_value(cosine_similarity, -1, 1)
#             # Normlize 0-1
#             cosine_dist = tf.acos(cosine_similarity) / math.pi
            
#             if cosine_dist.shape[0] == 1:
#                 eye = 1 - tf.eye(norm_feature.shape[0])
#                 masked_dist = cosine_dist * eye
#                 e_score = tf.reduce_sum(tf.reduce_min(masked_dist, axis=1))
#             else:
#                 # e_scores = torch.min(
#                     # cosine_dist + torch.eye(norm_feature.shape[0]).to(self.device),
#                     # -1)[0].sum()    # fixed bug
#                 eye = tf.eye(norm_feature.shape[0])
#                 adjusted_cosine_dist = cosine_dist + eye
#                 e_score = tf.reduce_sum(
#                     tf.reduce_min(adjusted_cosine_dist, axis=1))
#         return e_score # tensor(0.0343, device='cuda:0')
    
#     def proccess_compute_effectiveness_score(self, components_image_feature, combination_list):
#         """
#         Compute each S's effectiveness score
#         """
#         e_scores = []
#         for sub_index in combination_list:
#             sub_feature_set = tf.gather(components_image_feature, sub_index)    # shape=(batch, 1024)

#             e_score = self.compute_effectiveness_score(sub_feature_set)
#             e_scores.append(e_score.numpy())
        
#         return np.array(e_scores)
    
#     def merge_image(self, sub_index_set, partition_image_set, mode = "black"):
#         """
#         merge image
#         """
#         sub_image_set_ = np.array(partition_image_set)[sub_index_set]
#         if mode == "black":
#             image = sub_image_set_.sum(0)
#         elif mode == "gray":
#             image = sub_image_set_.sum(0)
#             image[image.sum(-1)==0] = 127

#         return image.astype(np.uint8)
    
#     def evaluation_maximun_sample(self, 
#                                   main_set, 
#                                   candidate_set, 
#                                   partition_image_set, 
#                                   monotonically_increasing):
#         """
#         Given a subset, return a best sample index
#         """
#         sub_index_sets = []
#         for candidate_ in candidate_set:
#             sub_index_sets.append(
#                 np.concatenate((main_set, np.array([candidate_]))).astype(int))

#         # Compute uncertainty / 计算不确定性
#         start = time.time()
#         # merge images / 组合图像
#         batch_input_images_u = np.array([
#             self.preprocess_image_uncertainty(
#                 self.merge_image(sub_index_set, partition_image_set)  # Uncertainty model is ONNX version
#             ) for sub_index_set in sub_index_sets])
        
#         u = self.compute_uncertainty(
#             batch_input_images_u
#         )
#         score_confidence = 1 - u
#         # print(score_confidence)
#         end = time.time()
#         # print('confidence程序执行时间: ',end - start)
        
#         # Compute Effectiveness Score / 计算有效性分数
#         start = time.time()
#         partition_image_features = np.array([
#             self.convert_prepare_image(
#                 partition_image
#             ) for partition_image in partition_image_set
#         ])

#         partition_image_features, _ = self.recognition_model(
#             partition_image_features
#         )
        
#         score_effectiveness = self.proccess_compute_effectiveness_score(
#             partition_image_features, sub_index_sets)
#         end = time.time()
#         # print('effectiveness程序执行时间: ',end - start)
        
#         # Compute Consistency Score 
#         start = time.time()
#         batch_input_images = np.array([
#             self.convert_prepare_image(
#                 self.merge_image(sub_index_set, partition_image_set)
#             ) for sub_index_set in sub_index_sets])
#         _, score_consistency = self.recognition_model(batch_input_images)
#         score_consistency = score_consistency.numpy()[:, self.target_label]
#         end = time.time()
#         # print('consistency程序执行时间: ',end - start)
        
#         # Compute Collaboration Score 
#         start = time.time()
#         batch_input_images_reverse = np.array([
#             self.convert_prepare_image(
#                 self.org_img - self.merge_image(sub_index_set, partition_image_set)
#             ) for sub_index_set in sub_index_sets])
#         _, score_collaboration = self.recognition_model(batch_input_images_reverse)
#         score_collaboration = 1 - score_collaboration.numpy()[:, self.target_label]
#         end = time.time()
#         # print('collaboration程序执行时间: ',end - start)

#         # submodular score
#         smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness +  self.lambda3 * score_consistency + self.lambda4 * score_collaboration
        
#         arg_max_index = smdl_score.argmax().item()
    
#         self.saved_json_file["confidence_score"].append(score_confidence[arg_max_index].item())
#         # self.saved_json_file["confidence_score"].append(score_confidence.item())
#         self.saved_json_file["effectiveness_score"].append(score_effectiveness[arg_max_index].item())
#         self.saved_json_file["consistency_score"].append(score_consistency[arg_max_index].item())
#         self.saved_json_file["collaboration_score"].append(score_collaboration[arg_max_index].item())
#         self.saved_json_file["smdl_score"].append(smdl_score[arg_max_index].item())

#         return sub_index_sets[arg_max_index]    # sub_index_sets is [main_set, new_candidate]
    
#     def get_merge_set(self, partition, monotonically_increasing = False):
#         """
#         """
#         Subset = np.array([])
        
#         indexes = np.arange(len(partition))
        
#         self.smdl_score_best = 0
        
#         for j in tqdm(range(self.k)):
#             # Sample a subsize of size s.
#             diff = np.setdiff1d(indexes, np.array(Subset))  # in indexes but not in Subset

#             sub_candidate_indexes = diff
            
#             Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition, monotonically_increasing)
            
#         return Subset
    
#     def __call__(self, image_set, id = None):
#         """
#         Compute Source Face Submodular Score
#             @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
#         """
#         self.saved_json_file = {}
#         self.saved_json_file["sub-k"] = self.k
#         self.saved_json_file["confidence_score"] = []
#         self.saved_json_file["effectiveness_score"] = []
#         self.saved_json_file["consistency_score"] = []
#         self.saved_json_file["collaboration_score"] = []
#         self.saved_json_file["smdl_score"] = []
#         self.saved_json_file["lambda1"] = self.lambda1
#         self.saved_json_file["lambda2"] = self.lambda2
#         self.saved_json_file["lambda3"] = self.lambda3
#         self.saved_json_file["lambda4"] = self.lambda4
        
#         self.org_img = np.array(image_set).sum(0).astype(np.uint8)
#         source_image = self.convert_prepare_image(
#                 self.org_img)
        
#         self.source_feature, predict = self.recognition_model(np.array([source_image]))

#         if id == None:
#             self.target_label = predict.numpy().argmax()
#         else:
#             self.target_label = id

#         Subset_merge = np.array(image_set)
#         # print(Subset_merge.shape)
#         # cv2.imwrite("Subset_merge.jpg", Subset_merge.sum(0))
#         Submodular_Subset = self.get_merge_set(     # array([30, 31,  1, ...])
#             Subset_merge, 
#             monotonically_increasing=True)

#         submodular_image_set = Subset_merge[Submodular_Subset]  # sub_k x (112, 112, 3)
        
#         submodular_image = submodular_image_set.sum(0).astype(np.uint8)

#         self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
#         self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
#         return submodular_image, submodular_image_set, self.saved_json_file

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch as AF
import math
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import models
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from collections import OrderedDict

import tensorflow_addons as tfa
from keras.models import load_model
from insight_face_models import *

import time

import torchvision.transforms as transforms
from .evidential import relu_evidence, exp_evidence

from tqdm import tqdm

class CubSubModularExplanationV2(object):
    def __init__(self, 
                 cfg_path="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
                 k=50,
                 lambda1=1.0,
                 lambda2=1.0,
                 lambda3=1.0,
                 lambda4=1.0):
        super(CubSubModularExplanationV2, self).__init__()
        
        # Load model configuration
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        assert self.cfg["version"] == 2
        
        self.device = tf.device("/CPU:0")  # Default to CPU; adjust if using GPU
        # self.device = torch.device(self.cfg["device"])
        # self.device = torch.device(self.cfg["device"])
        # self.device = self.cfg.get("device", "/CPU:0")
        self.moda = self.cfg["mode"]

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            
        # Initialize recognition model
        self.recognition_model = self.define_recognition_model(
            self.cfg["recognition_model"]["num_classes"], 
            self.cfg["recognition_model"]["model_path"]
        )

        # Parameters of the submodular
        self.k = k
        
        # Parameter of the LtLG algorithm
        self.ltl_log_ep = 5
        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

        self.softmax = tf.keras.layers.Softmax(axis=-1)

        if "resnet" in self.cfg["recognition_model"]["model_path"]:
            from tensorflow.keras.applications.resnet import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224
        elif "vgg19" in self.cfg["recognition_model"]["model_path"]:
            from tensorflow.keras.applications.vgg19 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224
        elif "efficientnetv2" in self.cfg["recognition_model"]["model_path"]:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 384
        elif "mobilenetv2" in self.cfg["recognition_model"]["model_path"]:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self.preprocess_input = preprocess_input
            self.tf_size = 224

    def convert_prepare_image(self, image, size=224):
        img = cv2.resize(image[..., ::-1], (self.tf_size, self.tf_size))
        img = self.preprocess_input(np.array(img))
        return img
    
    def preprocess_image_uncertainty(self, image, size=224):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = self.transforms(img).numpy()
        img = np.transpose(img, (1, 2, 0))  # Convert to (224, 224, 3) for TensorFlow
        return img

    # def define_recognition_model(self, num_classes, pretrained_path):
    #     """
    #     init the face recognition model
    #     """
    #     input_tensor = Input(shape=(224, 224, 3), name='input_1')
    #     base_model = ResNet101(weights='imagenet', include_top=False, 
    #                         input_tensor=input_tensor)
    #     # base_model = ResNet101(weights='imagenet', include_top=False, 
    #     #                       input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        
    #     # Add custom classification head
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(1024, activation='relu')(x)
    #     x = Dense(512, activation='relu')(x)
    #     class_number = 196
    #     predictions = Dense(class_number, activation='softmax')(x)
        
    #     # Create the final model
    #     model_base = Model(inputs=base_model.input, outputs=predictions)
    #     model_base.load_weights(pretrained_path)
    #     # model_base = load_model(pretrained_path)
    #     layer_name = "dense"
    #     # layer_name = "flatten"
    #     model = tf.keras.models.Model(inputs=model_base.input, outputs=[model_base.get_layer(layer_name).output, model_base.output])
    #     # model.layers[-1].activation = tf.keras.activations.linear
    #     print("Success load pre-trained bird recognition model {}".format(pretrained_path))

    #     return model

    # def define_recognition_model(self, num_classes, pretrained_path):
    #     """
    #     Initialize the recognition model with Laplace approximation using TensorFlow
    #     """
    #     input_tensor = Input(shape=(224, 224, 3), name='input_1')
    #     base_model = ResNet101(weights='imagenet', include_top=False, 
    #                         input_tensor=input_tensor)
        
    #     # Add feature extraction from base model output
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(1024, activation='relu')(x)
    #     x = Dense(512, activation='relu')(x)
    #     predictions = Dense(num_classes, activation='softmax')(x)
        
    #     model_base = Model(inputs=base_model.input, outputs=predictions)
    #     model_base.load_weights(pretrained_path)
    #     # model_base = load_model(pretrained_path)
    #     layer_name = "dense"
    #     # layer_name = "flatten"
    #     model = tf.keras.models.Model(inputs=model_base.input, outputs=[model_base.get_layer(layer_name).output, model_base.output])
    #     # model.layers[-1].activation = tf.keras.activations.linear
    #     print("Success load pre-trained bird recognition model {}".format(pretrained_path))

    #     return model
    #     # # Create the final model with multiple outputs
    #     # model_base = Model(inputs=base_model.input, outputs=[base_model.output, predictions])
    #     # model_base.load_weights(pretrained_path)
        
    #     # print("Success load pre-trained bird recognition model {}".format(pretrained_path))
    #     # return model_base
    
    def define_recognition_model(self, num_classes, pretrained_path):
        """
        Initialize the recognition model with Laplace approximation using TensorFlow
        and inspect the pretrained architecture.
        """
        input_tensor = Input(shape=(224, 224, 3), name='input_1')
        base_model = ResNet101(weights='imagenet', include_top=False, input_tensor=input_tensor)
        
        # Add feature extraction from base model output
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)  # Named as 'dense', not 'dense_2'
        # x = tf.clip_by_value(x, -50.0, 50.0, name='clipped_dense_input')
        predictions = Dense(9809, activation='softmax', name='predictions')(x)
        
        # Create the model with multiple outputs
        # model = tf.keras.models.Model(
        #     inputs=input_tensor,
        #     outputs=[base_model.get_layer('dense').output, predictions]  # Use 'dense' layer
        # )
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(pretrained_path)
        
        # Load weights into the full model
        print("Success load pre-trained bird recognition model {}".format(pretrained_path))
        
        # # Print layer details for verification
        # print("\nModel Architecture:")
        # for layer in model.layers:
        #     print(f"Layer: {layer.name}, Output Shape: {layer.output_shape}, Trainable: {layer.trainable}")
        
        # # Optionally, print weight shapes
        # for layer in model.layers:
        #     if layer.trainable_weights:
        #         for weight in layer.trainable_weights:
        #             print(f"Layer {layer.name}, Weight: {weight.name}, Shape: {weight.shape}")
                    
        model_multiple = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("dense").output, model.output])

        return model_multiple

    def exp_evidence(self, y):
        return np.exp(np.clip(y, -10, 10))
    
    # def compute_uncertainty(self, input_images, scale = 5):
    #     """
    #     Compute the uncertainty of the model
    #     input: torch.Size(batch, 3, w, h)
    #     """
    #     # print(input_images)
    #     # print(input_images.min(), input_images.max())
    #     with torch.no_grad():
    #         input_images = torch.from_numpy(input_images)
    #         output = self.uncertainty_model(input_images.to(self.device))
    #     evidence = exp_evidence(output)
    #     alpha = evidence + 1
    #     uncertainty = self.cfg["uncertainty_model"]["num_classes"] / torch.sum(alpha, dim=1, keepdim=True)

    #     return uncertainty.reshape(-1).cpu().numpy()
    # def compute_uncertainty(self, input_images, n_samples=5, scale=1.0):
    #     """
    #     Compute uncertainty by perturbing model weights and measuring gradient norms.
    #     Args:
    #         input_images (numpy.ndarray): Shape (batch, 3, H, W)
    #         n_samples (int): Number of perturbation samples
    #         scale (float): Scale factor for weight noise
    #     Returns:
    #         numpy.ndarray: Uncertainty values for each input in the batch
    #     """
    #     # Convert input to torch tensor and move to device
    #     input_tensor = torch.tensor(input_images, dtype=torch.float32).to(self.device)
    #     batch_size = input_tensor.size(0)
    #     uncertainties = np.zeros(batch_size)

    #     # Validate input tensor
    #     if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
    #         print("Input tensor contains NaN or Inf, aborting", flush=True)
    #         return np.zeros(batch_size)

    #     # Save original model weights
    #     # original_state_dict = {k: v.clone() for k, v in self.recognition_model.state_dict().items()}
    #     # For Keras model, use get_weights() instead of state_dict()
    #     original_weights = [w.copy() for w in self.recognition_model.get_weights()]

    #     def normalize_tensor(tensor, epsilon=1e-6):
    #         mean = tensor.mean(dim=[0], keepdim=True)
    #         variance = ((tensor - mean) ** 2).mean(dim=[0], keepdim=True)
    #         std = torch.sqrt(variance + epsilon)
    #         return (tensor - mean) / std

    #     for i in range(n_samples):
    #         # Restore original weights before perturbation
    #         self.recognition_model.load_state_dict(original_weights)

    #         # Perturb weights layer-wise
    #         for name, param in self.recognition_model.named_parameters():
    #             if param.requires_grad:
    #                 weight_magnitude = param.abs().mean()
    #                 random_factor = torch.empty(1).uniform_(0.1, 2.0).item()
    #                 noise_scale = scale * weight_magnitude * random_factor
    #                 std = torch.sqrt(((param - param.mean()) ** 2).mean())
    #                 noise = torch.randn_like(param) * std * noise_scale
    #                 new_param = param + noise
    #                 # Clip values to prevent explosion
    #                 clipping_range = 2.0 * max(param.min().abs(), param.max().abs())
    #                 new_param = torch.clamp(new_param, -clipping_range, clipping_range)
    #                 param.data.copy_(new_param)

    #         try:
    #             # Forward + gradient computation
    #             input_tensor.requires_grad_(True)
    #             output = self.recognition_model(input_tensor)
    #             if isinstance(output, (list, tuple)):
    #                 dense_output_final = output[0]
    #                 predictions = output[1]
    #             else:
    #                 dense_output_final = output
    #                 predictions = output

    #             loss = dense_output_final.mean()  # Dummy scalar for gradient computation
    #             gradients = torch.autograd.grad(outputs=loss, inputs=input_tensor,
    #                                             create_graph=False, retain_graph=False, only_inputs=True)[0]

    #             if gradients is None or torch.isnan(gradients).any():
    #                 print(f"Sample {i}: Invalid gradients, skipping", flush=True)
    #                 continue

    #             gradient_norm = torch.sqrt((gradients ** 2).mean(dim=[1, 2, 3]))
    #             uncertainties += gradient_norm.detach().cpu().numpy()

    #         except Exception as e:
    #             print(f"Sample {i} forward pass failed: {str(e)}", flush=True)
    #             continue
    #         finally:
    #             input_tensor.requires_grad_(False)

    #     # Restore original weights
    #     self.recognition_model.load_state_dict(original_weights)

    #     if n_samples > 0:
    #         uncertainties /= n_samples

    #     if np.all(uncertainties == 0):
    #         print("No valid samples produced, returning zeros", flush=True)
    #         return np.zeros(batch_size)
    #     if np.any(np.isnan(uncertainties)):
    #         print("NaN in uncertainties, replacing with 0", flush=True)
    #         uncertainties = np.nan_to_num(uncertainties, nan=0.0)

    #     print("Final uncertainties:", uncertainties, flush=True)
    #     return uncertainties

    def compute_uncertainty(self, input_images, n_samples=5, scale=1):
        input_tensor = tf.convert_to_tensor(input_images, dtype=tf.float32)
        batch_size = input_tensor.shape[0]
        uncertainties = np.zeros(batch_size)

        # Validate input tensor
        if tf.reduce_any(tf.math.is_nan(input_tensor)) or tf.reduce_any(tf.math.is_inf(input_tensor)):
            print("Input tensor contains NaN or Inf, aborting", flush=True)
            return np.zeros(batch_size)

        # Separate trainable and non-trainable weights
        original_trainable_weights = {}
        original_non_trainable_weights = {}
        for layer in self.recognition_model.layers:
            trainable_weights = layer.trainable_weights
            non_trainable_weights = layer.non_trainable_weights
            for var in trainable_weights:
                original_trainable_weights[var.name] = var.value().numpy()
            for var in non_trainable_weights:
                original_non_trainable_weights[var.name] = var.value().numpy()

        def normalize_tensor(tensor, epsilon=1e-6):
            mean = tf.reduce_mean(tensor, axis=[0], keepdims=True)  # Per-channel mean
            variance = tf.reduce_mean(tf.square(tensor - mean), axis=[0], keepdims=True)
            std = tf.sqrt(variance + epsilon)
            return (tensor - mean) / std

        for i in range(n_samples):
            # Restore all weights before perturbation
            for layer in self.recognition_model.layers:
                trainable_weights = layer.trainable_weights
                non_trainable_weights = layer.non_trainable_weights
                current_weights = [original_trainable_weights[var.name] for var in trainable_weights] + \
                                [original_non_trainable_weights[var.name] for var in non_trainable_weights]
                layer.set_weights(current_weights)

            # Perturb only trainable weights with layer-specific noise scale and dynamic std
            for layer in self.recognition_model.layers:
                trainable_weights = layer.trainable_weights
                current_weights = layer.get_weights()
                # Compute weight_magnitude only if trainable_weights exist
                if trainable_weights:
                    weight_magnitude = tf.reduce_mean([tf.reduce_mean(tf.abs(var.value())) for var in trainable_weights])
                else:
                    weight_magnitude = 1e-3  # Default small value for layers with no trainable weights
                # Add random variation per layer
                random_factor = tf.random.uniform([], minval=0.1, maxval=2.0, dtype=tf.float32)  # Wider range
                scale_tensor = tf.convert_to_tensor(scale * weight_magnitude * random_factor, dtype=tf.float32)
                # print(scale_tensor)
                noise_scale = scale_tensor
                # print(f"Sample {i}, Layer {layer.name}, Noise scale: {noise_scale.numpy()}", flush=True)
                for j, var in enumerate(trainable_weights):
                    weight_value = var.value()
                    # Compute std from weight variance, use fixed std for biases
                    mean_weight = tf.reduce_mean(weight_value)
                    variance_weight = tf.reduce_mean(tf.square(weight_value - mean_weight))
                    std = tf.sqrt(variance_weight + 1e-6) if j == 0 else tf.minimum(std, 0.01)  # Limit bias std
                    # print(f"Sample {i}, Layer {layer.name}, Weight {j}, Std: {std.numpy()}")
                    noise = tf.random.normal(weight_value.shape) * std * noise_scale
                    new_weight = weight_value + noise
                    if tf.reduce_any(tf.math.is_nan(new_weight)) or tf.reduce_any(tf.math.is_inf(new_weight)):
                        print(f"Sample {i}, Layer {layer.name}: NaN or Inf detected in new_weight, skipping perturbation", flush=True)
                        continue
                    weight_min = tf.reduce_min(weight_value)
                    weight_max = tf.reduce_max(weight_value)
                    clipping_range = tf.maximum(tf.abs(weight_min), tf.abs(weight_max)) * 2.0
                    new_weight = tf.clip_by_value(new_weight, -clipping_range, clipping_range)
                    current_weights[j] = new_weight.numpy()
                layer.set_weights(current_weights)

            try:
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(input_tensor)
                    output = self.recognition_model(input_tensor, training=False)
                    dense_output_final = output[0] if isinstance(output[0], tf.Tensor) else output[0][0]
                    predictions = output[1] if isinstance(output[1], tf.Tensor) else output[1][0]
                    
                    # Debug intermediate outputs using model layers
                    all_layer_outputs = [layer.output for layer in self.recognition_model.layers]
                    functional_model = tf.keras.models.Model(inputs=self.recognition_model.input, outputs=all_layer_outputs)
                    layer_outputs = functional_model(input_tensor)
                    
                    # Print outputs for key layers with normalization
                    resnet_output_layer = self.recognition_model.get_layer('conv5_block3_out')
                    if resnet_output_layer:
                        resnet_output = layer_outputs[self.recognition_model.layers.index(resnet_output_layer)]
                        resnet_output_normalized = normalize_tensor(resnet_output)
                        # print(f"Sample {i}, resnet_output shape: {resnet_output.shape}, normalized min: {tf.reduce_min(resnet_output_normalized).numpy()}, normalized max: {tf.reduce_max(resnet_output_normalized).numpy()}", flush=True)
                    else:
                        print(f"Sample {i}, No 'conv5_block3_out' layer found", flush=True)

                    pooled_output_layer = self.recognition_model.get_layer('global_average_pooling2d')
                    if pooled_output_layer:
                        pooled_output = layer_outputs[self.recognition_model.layers.index(pooled_output_layer)]
                        pooled_output_normalized = normalize_tensor(pooled_output)
                        # print(f"Sample {i}, pooled_output shape: {pooled_output.shape}, normalized min: {tf.reduce_min(pooled_output_normalized).numpy()}, normalized max: {tf.reduce_max(pooled_output_normalized).numpy()}", flush=True)

                    dense_1024_output_layer = self.recognition_model.get_layer('dense')
                    if dense_1024_output_layer:
                        dense_1024_output = layer_outputs[self.recognition_model.layers.index(dense_1024_output_layer)]
                        dense_1024_output_normalized = normalize_tensor(dense_1024_output)
                        # print(f"Sample {i}, dense_1024_output shape: {dense_1024_output.shape}, normalized min: {tf.reduce_min(dense_1024_output_normalized).numpy()}, normalized max: {tf.reduce_max(dense_1024_output_normalized).numpy()}", flush=True)

                    dense_output_layer = self.recognition_model.get_layer('dense_1')
                    if dense_output_layer:
                        dense_output = layer_outputs[self.recognition_model.layers.index(dense_output_layer)]
                        dense_output_normalized = normalize_tensor(dense_output)
                        # print(f"Sample {i}, dense_output shape: {dense_output.shape}, normalized min: {tf.reduce_min(dense_output_normalized).numpy()}, normalized max: {tf.reduce_max(dense_output_normalized).numpy()}", flush=True)

                    # print(f"Sample {i}, dense_output_final shape: {dense_output_final.shape}, normalized min: {tf.reduce_min(normalize_tensor(dense_output_final)).numpy()}, normalized max: {tf.reduce_max(normalize_tensor(dense_output_final)).numpy()}", flush=True)
                    # print(f"Sample {i}, predictions shape: {predictions.shape}, min: {tf.reduce_min(predictions).numpy()}, max: {tf.reduce_max(predictions).numpy()}", flush=True)
                gradients = tape.gradient(dense_output_final, input_tensor)
                if gradients is None:
                    print(f"Sample {i} gradients are None, likely due to nan in output", flush=True)
                    continue
                gradient_norm = tf.sqrt(tf.reduce_mean(tf.square(gradients), axis=[1, 2, 3]))
                # print(f"Sample {i}, Gradient norm: {gradient_norm.numpy()}", flush=True)
                if tf.reduce_any(tf.math.is_nan(gradient_norm)):
                    print(f"Sample {i} gradient_norm contains NaN, skipping", flush=True)
                    continue
                uncertainties += gradient_norm.numpy()
            except Exception as e:
                print(f"Sample {i} forward pass failed: {str(e)}", flush=True)
                continue
            finally:
                del tape

            # Restore all weights after each sample
            for layer in self.recognition_model.layers:
                trainable_weights = layer.trainable_weights
                non_trainable_weights = layer.non_trainable_weights
                current_weights = [original_trainable_weights[var.name] for var in trainable_weights] + \
                                [original_non_trainable_weights[var.name] for var in non_trainable_weights]
                layer.set_weights(current_weights)

        if n_samples > 0:
            uncertainties /= n_samples
        if np.all(uncertainties == 0):
            print("No valid samples produced, returning zeros", flush=True)
            return np.zeros(batch_size)
        if np.any(np.isnan(uncertainties)):
            print("NaN in uncertainties, replacing with 0", flush=True)
            uncertainties = np.nan_to_num(uncertainties, nan=0.0)

        print("Final uncertainties:", uncertainties, flush=True)
        return 10*uncertainties

    def compute_effectiveness_score(self, features):
        """
        Computes Effectiveness Score: The point should be distant from all the other elements in the subset.
        features: np.array(batch, d)
        """
        if self.cfg["effectiveness_distance_metric"] == "cosine":
            norm_feature = tf.nn.l2_normalize(features, axis=1)
            cosine_similarity = tf.matmul(norm_feature, tf.transpose(norm_feature))
            cosine_similarity = tf.clip_by_value(cosine_similarity, -1, 1)
            cosine_dist = tf.acos(cosine_similarity) / math.pi
            
            if cosine_dist.shape[0] == 1:
                eye = 1 - tf.eye(norm_feature.shape[0])
                masked_dist = cosine_dist * eye
                e_score = tf.reduce_sum(tf.reduce_min(masked_dist, axis=1))
            else:
                eye = tf.eye(norm_feature.shape[0])
                adjusted_cosine_dist = cosine_dist + eye
                e_score = tf.reduce_sum(tf.reduce_min(adjusted_cosine_dist, axis=1))
        return e_score.numpy()

    def proccess_compute_effectiveness_score(self, components_image_feature, combination_list):
        """
        Compute each S's effectiveness score
        """
        e_scores = []
        for sub_index in combination_list:
            sub_feature_set = tf.gather(components_image_feature, sub_index)
            e_score = self.compute_effectiveness_score(sub_feature_set.numpy())
            e_scores.append(e_score)
        return np.array(e_scores)

    def merge_image(self, sub_index_set, partition_image_set, mode="black"):
        """
        merge image
        """
        sub_image_set_ = np.array(partition_image_set)[sub_index_set]
        if mode == "black":
            image = sub_image_set_.sum(0)
        elif mode == "gray":
            image = sub_image_set_.sum(0)
            image[image.sum(-1)==0] = 127
        return image.astype(np.uint8)

    def evaluation_maximun_sample(self, 
                                main_set, 
                                candidate_set, 
                                partition_image_set, 
                                monotonically_increasing):
        """
        Given a subset, return a best sample index
        """
        sub_index_sets = []
        for candidate_ in candidate_set:
            sub_index_sets.append(
                np.concatenate((main_set, np.array([candidate_]))).astype(int))

        # Compute uncertainty
        start = time.time()
        batch_input_images_u = np.array([
            self.preprocess_image_uncertainty(
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        
        u = self.compute_uncertainty(batch_input_images_u)
        score_confidence = 1 - u
        end = time.time()
        
        # Compute Effectiveness Score
        start = time.time()
        partition_image_features = np.array([
            self.convert_prepare_image(partition_image)
            for partition_image in partition_image_set
        ])
        partition_image_features, _ = self.recognition_model.predict(partition_image_features)
        score_effectiveness = self.proccess_compute_effectiveness_score(
            partition_image_features, sub_index_sets)
        end = time.time()
        
        # Compute Consistency Score
        start = time.time()
        batch_input_images = np.array([
            self.convert_prepare_image(
                self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        _, score_consistency = self.recognition_model.predict(batch_input_images)
        score_consistency = score_consistency[:, self.target_label]
        end = time.time()
        
        # Compute Collaboration Score
        start = time.time()
        batch_input_images_reverse = np.array([
            self.convert_prepare_image(
                self.org_img - self.merge_image(sub_index_set, partition_image_set)
            ) for sub_index_set in sub_index_sets])
        _, score_collaboration = self.recognition_model.predict(batch_input_images_reverse)
        score_collaboration = 1 - score_collaboration[:, self.target_label]
        end = time.time()

        # submodular score
        smdl_score = self.lambda1 * score_confidence + self.lambda2 * score_effectiveness + self.lambda3 * score_consistency + self.lambda4 * score_collaboration
        
        arg_max_index = smdl_score.argmax()
    
        self.saved_json_file["confidence_score"].append(float(score_confidence[arg_max_index]))
        self.saved_json_file["effectiveness_score"].append(float(score_effectiveness[arg_max_index]))
        self.saved_json_file["consistency_score"].append(float(score_consistency[arg_max_index]))
        self.saved_json_file["collaboration_score"].append(float(score_collaboration[arg_max_index]))
        self.saved_json_file["smdl_score"].append(float(smdl_score[arg_max_index]))

        return sub_index_sets[arg_max_index]

    def get_merge_set(self, partition, monotonically_increasing=False):
        """
        """
        Subset = np.array([])
        
        indexes = np.arange(len(partition))
        
        self.smdl_score_best = 0
        
        for j in tqdm(range(self.k)):
            diff = np.setdiff1d(indexes, np.array(Subset))
            sub_candidate_indexes = diff
            
            Subset = self.evaluation_maximun_sample(Subset, sub_candidate_indexes, partition, monotonically_increasing)
            
        return Subset
    
    def __call__(self, image_set, id=None):
        """
        Compute Source Face Submodular Score
            @image_set: [mask_image 1, ..., mask_image m] (cv2 format)
        """
        self.saved_json_file = {}
        self.saved_json_file["sub-k"] = self.k
        self.saved_json_file["confidence_score"] = []
        self.saved_json_file["effectiveness_score"] = []
        self.saved_json_file["consistency_score"] = []
        self.saved_json_file["collaboration_score"] = []
        self.saved_json_file["smdl_score"] = []
        self.saved_json_file["lambda1"] = self.lambda1
        self.saved_json_file["lambda2"] = self.lambda2
        self.saved_json_file["lambda3"] = self.lambda3
        self.saved_json_file["lambda4"] = self.lambda4
        
        self.org_img = np.array(image_set).sum(0).astype(np.uint8)
        source_image = self.convert_prepare_image(self.org_img)
        
        self.source_feature, predict = self.recognition_model.predict(np.array([source_image]))
        if id == None:
            self.target_label = np.argmax(predict)
        else:
            self.target_label = id

        Subset_merge = np.array(image_set)
        Submodular_Subset = self.get_merge_set(Subset_merge, monotonically_increasing=True)

        submodular_image_set = Subset_merge[Submodular_Subset]
        submodular_image = submodular_image_set.sum(0).astype(np.uint8)

        self.saved_json_file["smdl_score_max"] = max(self.saved_json_file["smdl_score"])
        self.saved_json_file["smdl_score_max_index"] = self.saved_json_file["smdl_score"].index(self.saved_json_file["smdl_score_max"])
        
        return submodular_image, submodular_image_set, self.saved_json_file