

# 执行训练 Transfer AdaptSegnet
# load_path="checkpoints/20240517173008_baseline/temp.pth"
# python src/tools/transfer.py \
#     --config_file "configs/transfer.py" \
#     --load_path $load_path \
#     --tag "transfer"

# 执行训练 Transfer MCDDA
load_path="checkpoints/20240522181151_baseline/temp.pth"
python src/tools/transfer_mcdda.py \
    --config_file "configs/transfer_mcdda.py" \
    --load_path $load_path \
    --num_k 3 \
    --tag "Sun1toA3N05" \
    --merge_target_label True
