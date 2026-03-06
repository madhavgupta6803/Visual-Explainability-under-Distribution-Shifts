# Visual-Explainability-under-Distribution-Shifts
Uncertainty-Aware Subset Selection for Robust Visual Explainability under Distribution Shifts
https://openaccess.thecvf.com/content/WACV2026/papers/Gupta_Uncertainty-Aware_Subset_Selection_for_Robust_Visual_Explainability_under_Distribution_Shifts_WACV_2026_paper.pdf

# How to Run

## Generating the Explanations 
1. Download the checkpoints and put in the checkpoints folder
2. Go to smdl_explanation_cub.py file in the Submodular Attribution Folder
3. Update the dataset path in the arguments along with the eval-list path. Provide a suitable name in save-dir argumnent and run the file
4. After it is finished you will get a folder containing explanation files npy, jpg,etc.

## Evaluating the Explanations
1. Go to eval_AUC_faithfulness_atrribution.py
2. Update the path for the explanation directory in the parser argument
3. Run the file. You will be able to see the Insertion and Deletion AUC scores in the terminal
