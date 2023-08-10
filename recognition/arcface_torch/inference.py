import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    # inference函数有三个参数：权重、模型名称和图像。
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
    # 检查图像参数是否为 None，如果是，则随机生成一个 112x112 的三通道图像；
    # 如果不是，则使用 OpenCV 读取并调整大小为 112x112。

    # 将图像从 BGR 颜色空间转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 重新排列图像的维度顺序，将通道维度放在第一维。
    img = np.transpose(img, (2, 0, 1))

    # 将 Numpy 数组转换为 PyTorch 张量，并将其添加一个维度（用于代表批量大小），
    # 并将数据类型转换为浮点数。
    img = torch.from_numpy(img).unsqueeze(0).float()

    # 对图像进行规范化，使其像素值在 -1 和 1 之间。
    img.div_(255).sub_(0.5).div_(0.5)

    # 这三行代码加载 PyTorch 模型，并将其设置为评估模式。
    # 在这里，使用 get_model 函数根据给定的模型名称加载预训练的模型，然后使用 load_state_dict 方法加载权重参数。
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()

    # 使用 PyTorch 模型对输入图像进行特征提取，并将结果转换为 Numpy 数组。
    feat = net(img).numpy()

    # 输出提取的特征
    print(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='/mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth')   # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
