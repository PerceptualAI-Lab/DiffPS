CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/prw.yaml \
    --opts OUTPUT_DIR "DiffPS" \
    DATASET.BATCH_SIZE 3
