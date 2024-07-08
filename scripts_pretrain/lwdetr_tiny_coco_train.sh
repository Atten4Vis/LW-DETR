model_name='lwdetr_tiny_objects365'
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
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 1.0 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 6 \
    --window_block_indexes 0 2 4 \
    --out_feature_indexes 1 3 5 \
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
    --num_queries 100 \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 100 \
    --dataset_file o365 \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --use_ema \
    --pretrained_encoder pretrain_weights/caev2_tiny_300e_objects365.pth \
    --output_dir output/$model_name
