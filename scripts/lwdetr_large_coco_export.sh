model_name='lwdetr_large_coco'
coco_path=$1
checkpoint=$2

python main.py \
    --batch_size 1 \
    --encoder vit_small \
    --vit_encoder_num_layers 10 \
    --window_block_indexes 0 1 3 6 7 9 \
    --out_feature_indexes 2 4 5 9 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P3 P5 \
    --hidden_dim 384 \
    --sa_nheads 12 \
    --ca_nheads 24 \
    --dec_n_points 4 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_select 300 \
    --dataset_file coco \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --use_ema \
    --eval \
    --resume $checkpoint \
    --output_dir output/$model_name \
    export_model ${@:3}