model_name='lwdetr_small_coco'
coco_path=$1

python -u -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 4 \
    --weight_decay 1e-4 \
    --epochs 60 \
    --lr_drop 60 \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
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
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 300 \
    --dataset_file coco \
    --coco_path $coco_path \
    --square_resize_div_64 \
    --use_ema \
    --pretrained_encoder pretrain_weights/caev2_tiny_300e_objects365.pth \
    --pretrain_weights pretrain_weights/LWDETR_small_30e_objects365.pth \
    --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.weight transformer.enc_out_class_embed.1.weight transformer.enc_out_class_embed.2.weight transformer.enc_out_class_embed.3.weight transformer.enc_out_class_embed.4.weight transformer.enc_out_class_embed.5.weight transformer.enc_out_class_embed.6.weight transformer.enc_out_class_embed.7.weight transformer.enc_out_class_embed.8.weight transformer.enc_out_class_embed.9.weight transformer.enc_out_class_embed.10.weight transformer.enc_out_class_embed.11.weight transformer.enc_out_class_embed.12.weight transformer.enc_out_class_embed.0.bias transformer.enc_out_class_embed.1.bias transformer.enc_out_class_embed.2.bias transformer.enc_out_class_embed.3.bias transformer.enc_out_class_embed.4.bias transformer.enc_out_class_embed.5.bias transformer.enc_out_class_embed.6.bias transformer.enc_out_class_embed.7.bias transformer.enc_out_class_embed.8.bias transformer.enc_out_class_embed.9.bias transformer.enc_out_class_embed.10.bias transformer.enc_out_class_embed.11.bias transformer.enc_out_class_embed.12.bias class_embed.weight class_embed.bias \
    --output_dir output/$model_name
