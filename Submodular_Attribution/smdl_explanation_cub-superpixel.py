import argparse
import os
import numpy as np
import cv2
import json
import imageio
from matplotlib import pyplot as plt
from sklearn import metrics
plt.style.use('seaborn')

from tqdm import tqdm
from utils import *
import time

from models.submodular_cub_v2 import CubSubModularExplanationV2

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/NABirds/test',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/NABirds/image_paths.txt',
                        help='Datasets.')
    parser.add_argument('--division',
                        type=str,
                        default="superpixel",
                        choices=["grad", "pixel", "superpixel"],
                        help="")
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="seeds",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--sub-n', 
                        type=int, default=1,
                        help='')
    parser.add_argument('--sub-k', 
                        type=int, default=24,
                        help='')
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda3', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda4', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--cfg', 
                        type=str, 
                        default="configs/cub/submodular_cfg_cub_tf-resnet-v2.json",
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='submodular_results_NABirds_Superpixel_seeds/NABirds',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def SubRegionDivision(image, mode="slico"):
    element_sets_V = []
    if mode == "slico":
        slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=30, ruler=20.0) 
        slic.iterate(20)
        label_slic = slic.getLabels()
        number_slic = slic.getNumberOfSuperpixels()

        for i in range(number_slic):
            img_copp = image.copy()
            img_copp = img_copp * (label_slic == i)[:, :, np.newaxis]
            element_sets_V.append(img_copp)
    elif mode == "seeds":
        seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], num_superpixels=50, num_levels=3)
        seeds.iterate(image, 10)
        label_seeds = seeds.getLabels()
        number_seeds = seeds.getNumberOfSuperpixels()

        for i in range(number_seeds):
            img_copp = image.copy()
            img_copp = img_copp * (label_seeds == i)[:, :, np.newaxis]
            element_sets_V.append(img_copp)
    return element_sets_V

def visualization(image, submodular_image_set, saved_json_file, save_path):
    insertion_ours_images = []
    if len(submodular_image_set) == 0:
        print(f"Warning: Empty submodular_image_set for {save_path}")
        return
    insertion_image = submodular_image_set[0]
    insertion_ours_images.append(insertion_image)
    for smdl_sub_mask in submodular_image_set[1:]:
        insertion_image = insertion_image.copy() + smdl_sub_mask
        insertion_ours_images.append(insertion_image)
    insertion_ours_images_input_results = np.array(saved_json_file.get("consistency_score", []))
    if len(insertion_ours_images_input_results) == 0:
        print(f"Warning: Empty consistency_score for {save_path}")
        return
    ours_best_index = next((i for i, score in enumerate(insertion_ours_images_input_results) if score > 0.85), np.argmax(insertion_ours_images_input_results))
    x = [(insertion_ours_image.sum(-1)!=0).sum() / (image.shape[0] * image.shape[1]) for insertion_ours_image in insertion_ours_images]
    fig, [ax2, ax3] = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1, 1.5]}, figsize=(24, 8))
    ax2.axis('off')
    ax2.set_title('Ours', fontsize=54)
    ax3.set_xlim((0, 1))
    ax3.set_ylim((0, 1))
    ax3.tick_params(axis='both', labelsize=36)
    ax3.set_title('Insertion', fontsize=54)
    ax3.set_ylabel('Recognition Score', fontsize=44)
    ax3.set_xlabel('Percentage of image revealed', fontsize=44)
    ours_y = insertion_ours_images_input_results[:len(x)]
    ax3.plot(x, ours_y, color='dodgerblue', linewidth=3.5)
    ax3.scatter(x[-1], ours_y[-1], color='dodgerblue', s=54)
    kernel = np.ones((3, 3), dtype=np.uint8)
    ax3.plot([x[ours_best_index], x[ours_best_index]], [0, 1], color='red', linewidth=3.5)
    mask = (image - insertion_ours_images[ours_best_index]).mean(-1)
    mask[mask>0] = 1
    dilate = cv2.dilate(mask, kernel, iterations=3)
    edge = dilate - mask
    image_debug = image.copy()
    image_debug[mask>0] = image_debug[mask>0] * 0.5
    image_debug[edge>0] = np.array([0, 0, 255])
    ax2.imshow(image_debug[..., ::-1])
    auc_score = metrics.auc(x, ours_y)
    print(f"Highest confidence: {ours_y.max()}\nFinal confidence: {ours_y[-1]}\nInsertion AUC: {auc_score}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def main(args):
    smdl = CubSubModularExplanationV2(cfg_path=args.cfg, k=args.sub_k, 
                                      lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3, lambda4=args.lambda4)
    
    with open(args.eval_list, "r") as f:
        infos = f.read().split('\n')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.division == "superpixel":
        save_dir = os.path.join(args.save_dir, "{}-{}-{}-{}-{}-{}".format(
            args.division, args.superpixel_algorithm, args.lambda1, args.lambda2, args.lambda3, args.lambda4
        ))
    
    os.makedirs(save_dir, exist_ok=True)
    
    for info in tqdm(infos[:]):
        if not info.strip():
            continue
        id_people = info.split(" ")[-1]
        image_relative_path = info.split(" ")[0]
        
        image_path = os.path.join(args.Datasets, image_relative_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        
        element_sets_V = SubRegionDivision(image, mode=args.superpixel_algorithm)
        smdl.k = len(element_sets_V)

        start = time.time()
        submodular_image, submodular_image_set, saved_json_file = smdl(element_sets_V)
        end = time.time()
        print(f"Execution time: {end - start:.2f} seconds")
        
        # Save image
        save_image_root_path = os.path.join(save_dir, "image")
        os.makedirs(os.path.join(save_image_root_path, id_people), exist_ok=True)
        cv2.imwrite(os.path.join(save_image_root_path, image_relative_path), submodular_image)

        # Save npy
        save_npy_root_path = os.path.join(save_dir, "npy")
        os.makedirs(os.path.join(save_npy_root_path, id_people), exist_ok=True)
        np.save(os.path.join(save_npy_root_path, image_relative_path.replace(".jpg", ".npy")),
                np.array(submodular_image_set))

        # Save json
        save_json_root_path = os.path.join(save_dir, "json")
        os.makedirs(os.path.join(save_json_root_path, id_people), exist_ok=True)
        with open(os.path.join(save_json_root_path, image_relative_path.replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

        # Save GIF (optional)
        save_gif_root_path = os.path.join(save_dir, "gif")
        os.makedirs(os.path.join(save_gif_root_path, id_people), exist_ok=True)
        
        # Save visualization
        save_vis_root_path = os.path.join(save_dir, "visualization")
        os.makedirs(os.path.join(save_vis_root_path, id_people), exist_ok=True)
        vis_save_path = os.path.join(save_vis_root_path, image_relative_path.replace(".jpg", ".png"))
        visualization(image, submodular_image_set, saved_json_file, vis_save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
