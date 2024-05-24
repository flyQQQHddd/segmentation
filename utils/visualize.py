
from PIL import Image
from argparse import ArgumentParser
import numpy as np
Image.MAX_IMAGE_PIXELS = 1000000000000
THUMBNAIL_MAX_SIZE = (3000, 3000) # 缩略图最大大小

if __name__ == "__main__":

    # 获取命令行参数
    parser = ArgumentParser(description="")
    parser.add_argument("--input",type=str)
    parser.add_argument("--output",type=str)
    arg = parser.parse_args()
    input = arg.input
    output = arg.output

    # 输出参数信息
    print("Input:", input)
    print("Output:", output)

    # 读取图像
    img = Image.open(input)

    # 输出基本信息
    width, height = img.size
    prange = img.getextrema()
    print("Image size:", img.size)
    print("Image pixel range:", prange)

    # 转为 numpy 数组
    img = np.array(img)

    # 颜色映射表
    color_map = np.array([
        [0, 0, 0],   # Black
        [0, 255, 0], # Green
        [205,133,63] # Brown
    ]) 

    # 色彩映射
    colored_image = color_map[img].astype(np.uint8)

    # 保存图像
    img = Image.fromarray(colored_image)
    img.save(output)

    # 生成缩略图
    img.thumbnail(THUMBNAIL_MAX_SIZE)
    img.save(output.replace(".png", "_thumbnail.png"))








