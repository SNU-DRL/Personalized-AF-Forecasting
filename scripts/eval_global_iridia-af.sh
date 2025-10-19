# Dataset
TRAIN_DATA_PATH=/data4/ecg_data/iridia-af/dataset_final/test_p_tr.pickle
EVAL_DATA_PATH=/data4/ecg_data/iridia-af/dataset_final/test_p_te.pickle

# Pre-trained model
EXP_NAME=resnet18_15_bs128_lr5e-3_wd1e-4_ep3
MODEL_BASE_PATH=results_global/$EXP_NAME

# Settings
GPU_NUM=7

mkdir results_personalized_iridia-af

RESULT_DIR='./results_personalized_iridia-af/'$EXP_NAME'_baseline'
for SEED in 0 1 2 3 4
do
    MODEL_LOAD_PATH=$MODEL_BASE_PATH'_seed'$SEED'/checkpoints/model_last.pt'
    RESULT_SEED_DIR=$RESULT_DIR'_seed'$SEED

    python main_personalized.py \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --model_load_path $MODEL_LOAD_PATH \
        --mode evaluation \
        --gpu_num $GPU_NUM \
        --seed $SEED \
        --result_dir $RESULT_SEED_DIR
done

python analysis/process_metrics.py $RESULT_DIR
