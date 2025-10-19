# Dataset
EVAL_DATA_PATH=/data4/ecg_data/icentia11k/dataset_final/test_p_te.pickle

# Pre-trained model
EXP_NAME=resnet18_15_bs128_lr5e-3_wd1e-4_ep3__bs32_lr1e-3_wd1e-4_ep5_seed0
MODEL_PATH=results_personalized_icentia11k/$EXP_NAME

# Settings
GPU_NUM=0

mkdir results_attribution_icentia11k

RESULT_DIR='results_attribution_icentia11k/'$EXP_NAME

python main_personalized_attribution.py \
    --eval_data_path $EVAL_DATA_PATH \
    --model_load_dir $MODEL_PATH \
    --gpu_num $GPU_NUM \
    --result_dir $RESULT_DIR
