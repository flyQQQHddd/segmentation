

# 执行推理
model="checkpoints/20240524191746_A5N05/temp.pth"
config_file="configs/inference.py"
# python src/tools/inference.py \
#     --config_file $config_file \
#     --load_path $model \
#     --tag A5N05


# 执行推理 MCDDA
config_file="configs/inference_mcdda.py"
model_g="checkpoints/20240523182022_Sun1toA3N20/15_G.pth"
model_f="checkpoints/20240523182022_Sun1toA3N20/15_F1.pth"
# python src/tools/inference_mcdda.py \
#     --config_file $config_file \
#     --load_path_G $model_g \
#     --load_path_F $model_f \
#     --tag Sun1toA3N20


# Visualize
path="/home/featurize/work/code/out/20240524191925_A5N20/"
python utils/visualize.py \
    --input $path"output.png" \
    --output $path"visual.png"

path="/home/featurize/work/code/out/20240524192453_A5N15/"
python utils/visualize.py \
    --input $path"output.png" \
    --output $path"visual.png"

path="/home/featurize/work/code/out/20240524194428_A5N10/"
python utils/visualize.py \
    --input $path"output.png" \
    --output $path"visual.png"

path="/home/featurize/work/code/out/20240524195219_A5N05/"
python utils/visualize.py \
    --input $path"output.png" \
    --output $path"visual.png"



