# Dataset
TRAIN_DATA_PATH=/data4/ecg_data/icentia11k/dataset_final/training.pickle
EVAL_DATA_PATH=/data4/ecg_data/icentia11k/dataset_final/validation.pickle

# Model
ARCH="resnet18_15"

# Hyperparameters
BATCH_SIZE=128
LEARNING_RATE=1e-1
WEIGHT_DECAY=1e-4
EPOCHS=5

# Settings
GPU_NUM=0

mkdir results_global

RESULT_DIR='./results_global/'$ARCH'_bs'$BATCH_SIZE'_lr'$LEARNING_RATE'_wd'$WEIGHT_DECAY'_ep'$EPOCHS
for SEED in 0 1 2 3 4
do
    RESULT_SEED_DIR=$RESULT_DIR'_seed'$SEED

    python main.py \
        --train_data_path $TRAIN_DATA_PATH \
        --eval_data_path $EVAL_DATA_PATH \
        --arch $ARCH \
        -bs $BATCH_SIZE \
        -lr $LEARNING_RATE \
        -wd $WEIGHT_DECAY \
        -ep $EPOCHS \
        --mode fit \
        --gpu_num $GPU_NUM \
        --seed $SEED \
        --result_dir $RESULT_SEED_DIR \
        --include_af_episodes \
        --use_augmentation
done

python analysis/process_metrics.py $RESULT_DIR
