model_name='lwdetr_small_coco'
checkpoint=$1
image_path=$2
output_dir=$3

python demo/demo.py \
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
    --weights $checkpoint \
    --input $image_path \
    --output $output_dir/$model_name \
    --confidence_threshold 0.5
