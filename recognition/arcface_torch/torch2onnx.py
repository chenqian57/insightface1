import numpy as np
import onnx
import torch
# from onnxsim import simplify

# 要输入一个PyTorch模型（net）、模型参数的路径（path_module）、输出ONNX模型的路径（output）、ONNX版本（opset）以及一个布尔值参数simplify。
# opset默认为 11，
def convert_onnx(net, path_module, output, opset=15, simplify=False):
    assert isinstance(net, torch.nn.Module)

    # 函数生成一个随机的三通道图像，并将其归一化处理，最后将其转换为PyTorch张量
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    # img = img.astype(np.float)  
    # img = img.astype(float)

    img = img.astype(np.float64)


    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    weight = torch.load(path_module)

    # 通过torch.load从path_module中加载模型参数，并将其加载到net中
    net.load_state_dict(weight, strict=True)
    # net的评估模式设置为eval
    net.eval()

    # torch.onnx.export将net模型转换为ONNX模型。
    # 其中，img是模型的输入数据，output是保存ONNX模型的文件路径，
    # input_names是模型输入的名称，keep_initializers_as_inputs参数表示是否保留模型中的初始参数，opset_version表示要使用的ONNX版本。
    torch.onnx.export(net, img, output, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    # backbone.pth
    # data


    # 加载导出的 ONNX 模型，并获取模型图
    model = onnx.load(output)
    graph = model.graph

    # 修改模型图中的输入维度，将其设置为动态维度
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    # 如果simplify参数为True，则使用 ONNXSIM库 的 simplify函数 简化模型，同时检查简化后的模型是否合法。最后将简化后的模型保存到输出文件中。
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)



    
if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model
    # r50


    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')

    # parser.add_argument方法用于添加命令行参数，包括输入文件路径、输出文件路径、网络类型以及是否对ONNX模型进行简化等参数
    parser.add_argument('--input', type=str,default="/mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth", help='input backbone.pth file or path')
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r34_fp16/backbone.pth
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth
    # /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r100_fp16/backbone.pth

    parser.add_argument('--output', type=str, default="/mnt/ssd/qiujing/arcface/megafacedata/conver1/model_r50v2.onnx", help='output onnx path')
    # /mnt/ssd/qiujing/arcface/megafacedata/conver1

    parser.add_argument('--network', type=str, default="r50", help='backbone network')

    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input

    # 如果输入文件是一个目录，则将其与默认文件名model.pt拼接，得到输入文件的完整路径
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "backbone.pth")     # "model.pt"

    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    assert args.network is not None
    print(args)

    # 创建一个backbone_onnx对象，它是通过调用get_model函数根据网络类型创建的一个PyTorch模型
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model_r50.onnx")
    
    # 最后，调用convert_onnx函数将PyTorch模型转换为ONNX模型，并将其保存到输出文件中。
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify)
