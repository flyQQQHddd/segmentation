
# -------------------------------------------------
# 数据集保存路径
save_dir="/home/featurize/data/WHUSDataset"
# 数据集路径
dataset="/home/featurize/data/WHUS/"
# 处理数据（标签和数据）
_8cm_sun1=$dataset"image/JianLi_DOM_8cm_sun1.png"
_8cm_sun1_label=$dataset"label/JianLi_DOM_8cm_sun1.png"
_8cm_sun2=$dataset"image/JianLi_DOM_8cm_sun2.png"
_8cm_sun2_label=$dataset"label/JianLi_DOM_8cm_sun2.png"
_8cm_cloud1=$dataset"image/JianLi_DOM_8cm_cloud1.png"
_8cm_cloud1_label=$dataset"label/JianLi_DOM_8cm_cloud1.png"
_8cm_cloud2=$dataset"image/JianLi_DOM_8cm_cloud2.png"
_8cm_cloud2_label=$dataset"label/JianLi_DOM_8cm_cloud2.png"
_8cm_after1=$dataset"image/JianLi_DOM_8cm_after1.png"
_8cm_after1_label=$dataset"label/JianLi_DOM_8cm_after1.png"
_8cm_after2=$dataset"image/JianLi_DOM_8cm_after2.png"
_8cm_after2_label=$dataset"label/JianLi_DOM_8cm_after2.png"
_part=$dataset"image/part.png"
_cropland=$dataset"image/cropland.png"
_cropland_label=$dataset"label/cropland.png"

# 执行数据切割
# python utils/split.py -image_path $_8cm_sun1 -label_path $_8cm_sun1_label -save_dir $save_dir
# python utils/split.py -image_path $_8cm_sun2 -label_path $_8cm_sun2_label -save_dir $save_dir
# python utils/split.py -image_path $_8cm_cloud1 -label_path $_8cm_cloud1_label -save_dir $save_dir
# python utils/split.py -image_path $_8cm_cloud2 -label_path $_8cm_cloud2_label -save_dir $save_dir
# python utils/split.py -image_path $_8cm_after1 -label_path $_8cm_after1_label -save_dir $save_dir
# python utils/split.py -image_path $_8cm_after2 -label_path $_8cm_after2_label -save_dir $save_dir
# python utils/split.py -image_path $_part -save_dir $save_dir
# python utils/split.py -image_path $_cropland -label_path $_cropland_label -save_dir $save_dir


# -------------------------------------------------
# 数据集保存路径
save_dir="/home/featurize/data/TransferDataset"
# 数据集路径
dataset="/home/featurize/data/Hainan/"
# 处理数据（标签和数据）
Hainan_A3_4cm_test=$dataset"image/Hainan_A3_4cm_test.png"
Hainan_A3_4cm_train=$dataset"image/Hainan_A3_4cm_train.png"
Hainan_A5_2cm_test=$dataset"image/Hainan_A5_2cm_test.png"
Hainan_A5_2cm_train=$dataset"image/Hainan_A5_2cm_train.png"
Jianli_8cm=$dataset"image/Jianli_8cm.png"
Hainan_A3_4cm_test_label=$dataset"label/Hainan_A3_4cm_test.png"
Hainan_A3_4cm_train_label=$dataset"label/Hainan_A3_4cm_train.png"
Hainan_A5_2cm_test_label=$dataset"label/Hainan_A5_2cm_test.png"
Hainan_A5_2cm_train_label=$dataset"label/Hainan_A5_2cm_train.png"
Jianli_8cm_label=$dataset"label/Jianli_8cm.png"
# 执行数据切割
python utils/split.py -image_path $Hainan_A3_4cm_test -label_path $Hainan_A3_4cm_test_label -save_dir $save_dir
python utils/split.py -image_path $Hainan_A3_4cm_train -label_path $Hainan_A3_4cm_train_label -save_dir $save_dir
python utils/split.py -image_path $Hainan_A5_2cm_test -label_path $Hainan_A5_2cm_test_label -save_dir $save_dir
python utils/split.py -image_path $Hainan_A5_2cm_train -label_path $Hainan_A5_2cm_train_label -save_dir $save_dir
python utils/split.py -image_path $Jianli_8cm -label_path $Jianli_8cm_label -save_dir $save_dir



