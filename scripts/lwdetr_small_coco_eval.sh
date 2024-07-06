model_name='lwdetr_small_coco'
coco_path=$1
checkpoint=$2

python -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --batch_size 4 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 10 \
    --window_block_indexes 0 1 3 6 7 9 \
    --out_feature_indexes 2 4 5 9 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_select 300 \
    --dataset_file coco \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --use_ema \
    --eval --resume $checkpoint \
    --output_dir output/$model_name