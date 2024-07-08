model_name='lwdetr_large_objects365'
coco_path=$1

python -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --lr 4e-4 \
    --lr_encoder 6e-4 \
    --batch_size 4 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --lr_drop 30 \
    --lr_vit_layer_decay 0.7 \
    --lr_component_decay 1.0 \
    --encoder vit_small \
    --drop_path 0.05 \
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
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 300 \
    --dataset_file o365 \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --use_ema \
    --pretrained_encoder pretrain_weights/caev2_small_300e_objects365.pth \
    --output_dir output/$model_name
