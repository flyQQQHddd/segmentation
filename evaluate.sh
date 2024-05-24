
# baseline
label='/home/featurize/data/WHUS/label/cropland.png'
predict='/home/featurize/work/code/out/20240518202518_mcdda_cropland/output.png'
python ./src/tools/evaluate.py -p $predict -l $label
