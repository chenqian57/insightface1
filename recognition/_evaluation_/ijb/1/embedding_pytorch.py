import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
from skimage import transform as trans

import sys
sys.path.append('/home/qiujing/cqwork/insightface/recognition/_evaluation_/ijb/1')
from arcface import get_model

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), 
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), 
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                          # Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=False))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           #Dropout(0.4),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        conv_out = x.view(x.shape[0], -1)
        x = self.output_layer(x)

        return x, conv_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_100(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model










from torch.nn.functional import normalize
from tqdm import tqdm


# 包含了一个初始化函数和一个get函数。
# 初始化函数中，首先定义了一个输入图片的大小，然后将设备设置为cuda:0，
# 接着调用了get_model函数来获取模型，将模型放到设备上，并加载了预训练模型的权重。
# 接下来定义了RGB_MEAN和RGB_STD，以及一个图像变换的transform。
# 最后定义了一个src数组和图像大小。
# get函数中，首先判断landmark的形状是否正确，然后根据landmark的形状来计算landmark5。
# 接着计算仿射变换矩阵M，将输入图像进行仿射变换，然后进行RGB颜色空间的转换。
# 接下来对图像进行翻转，然后进行图像变换，最后将变换后的图像输入到模型中，得到特征向量feat，将其展平并返回。

# class Embedding:

#   def __init__(self, imgs, backbone, prefix, epoch, ctx_id=0):
#       input_size = [112,112]
#       self.device = torch.device("cuda:{}".format(ctx_id))

#       self.model = get_model(backbone, fp16=False).to(self.device)
#       # self.model = IR_50(input_size).to(self.device)


#       state_dict = torch.load(prefix, map_location=self.device)
#       self.model.load_state_dict(state_dict)
#       self.model.eval()



#       for idx, img in enumerate(imgs):
#           img = img[:,:,::-1] #to rgb
#           img = img.transpose(2, 0, 1)
#         # img = np.transpose(img, (2,0,1))

#           img = np.ascontiguousarray(img)
#           img = torch.from_numpy(img)
#           img = img.clone().detach()
#          # data[count+idx] = img


#       embeddings = []
#       img = img.cuda()
#       img = img.float()
#       img.div_(255).sub_(0.5).div_(0.5)
#       img = img.unsqueeze(0)
#       # print(img.shape)
#       # 将数据传入模型中进行预测
#       out = self.model(img)
#       out = normalize(out)

#         # tensor张量对象
#         # out = torch.nn.functional.normalize(out)
        
#         # 把 out 转为 numpy 数组
#       embedding = out.detach().cpu().numpy()
#       embeddings.append(embedding)
#         # pbar.set_description(s)



#       embeddings = np.concatenate(embeddings, axis=0)
#       return embeddings






#       RGB_MEAN = [0.5, 0.5, 0.5]
#       RGB_STD = [0.5, 0.5, 0.5]

#       self.transform = transforms.Compose([
#            transforms.ToPILImage(),
#            transforms.ToTensor(), 
#            transforms.Normalize(mean = RGB_MEAN,
#                             std = RGB_STD),])
#       src = np.array([
#         [30.2946, 51.6963],
#         [65.5318, 51.5014],
#         [48.0252, 71.7366],
#         [33.5493, 92.3655],
#         [62.7299, 92.2041] ], dtype=np.float32 )
#       src[:,0] += 8.0
#       self.src = src
#       self.image_size = input_size

#   def get(self, rimg, landmark):
#     assert landmark.shape[0]==68 or landmark.shape[0]==5
#     assert landmark.shape[1]==2
#     if landmark.shape[0]==68:
#       landmark5 = np.zeros( (5,2), dtype=np.float32 )
#       landmark5[0] = (landmark[36]+landmark[39])/2
#       landmark5[1] = (landmark[42]+landmark[45])/2
#       landmark5[2] = landmark[30]
#       landmark5[3] = landmark[48]
#       landmark5[4] = landmark[54]
#     else:
#       landmark5 = landmark
#     tform = trans.SimilarityTransform()
#     tform.estimate(landmark5, self.src)
#     M = tform.params[0:2,:]
#     img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_flip = np.fliplr(img)
#     #img = np.transpose(img, (2,0,1)) #3*112*112, RGB
#     #img_flip = np.transpose(img_flip,(2,0,1))
#     #print(img.shape, img_flip.shape)
#     sample = self.transform(img)
#     sample_flip = self.transform(img_flip)
#     #print(sample.shape, sample_flip.shape)
#     sample_input = torch.stack([sample,sample_flip],0).to(self.device)
#     #print(sample_input.shape)
#     with torch.no_grad():
#         feat = self.model.forward(sample_input)[0].cpu().numpy()
#     #print(feat.shape)
#     feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])

#     feat = feat.flatten()
#     return feat








import mxnet as mx






class Embedding:
  # 初始化人脸识别模型
  # ctx_id表示 GPU设备编号
  def __init__(self, backbone, prefix, epoch, ctx_id=0):

      input_size = [112,112]
      self.device = torch.device("cuda:{}".format(ctx_id))

      self.model = get_model(backbone, fp16=False).to(self.device)

      state_dict = torch.load(prefix, map_location=self.device)
      self.model.load_state_dict(state_dict)
      self.model.eval()




    #   # 用于图像变换的参数
    #   RGB_MEAN = [0.5, 0.5, 0.5]
    #   RGB_STD = [0.5, 0.5, 0.5]
    #   # 图像变换transform
    #   self.transform = transforms.Compose([
    #        transforms.ToPILImage(),
    #        transforms.ToTensor(), 
    #        transforms.Normalize(mean = RGB_MEAN,
    #                         std = RGB_STD),])
      
    #   # 定义src数组和图像大小
    #   # src数组是人脸对齐的关键点坐标
    #   src = np.array([
    #     [30.2946, 51.6963],
    #     [65.5318, 51.5014],
    #     [48.0252, 71.7366],
    #     [33.5493, 92.3655],
    #     [62.7299, 92.2041] ], dtype=np.float32 )
    #   src[:,0] += 8.0
    #   self.src = src
      self.image_size = input_size


  # 获取图像的特征向量
  # 将输入的图像进行对齐、变换和特征提取，并返回特征向量
  # rimg表示输入的图像，landmark表示输入图像中人脸的关键点坐标
  def get(self, rimg):
        print (rimg.shape)
        if rimg.shape[-2:] != self.image_size:
            rimg = cv2.resize(rimg, (self.image_size[1], self.image_size[0]))
            # rimg = mx.image.resize_short(rimg, self.image_size)
        

        rimg = rimg[:,:,::-1] #to rgb
        rimg = rimg.transpose(2, 0, 1)
        # img = np.transpose(img, (2,0,1))

        rimg = np.ascontiguousarray(rimg)
        rimg = torch.from_numpy(rimg)
        # 创建一个张量的副本，并将其从计算图中分离
        rimg = rimg.clone().detach()


        rimg = rimg.cuda()
        rimg = rimg.float()
        rimg.div_(255).sub_(0.5).div_(0.5)

        rimg = rimg.unsqueeze(0)
        # print(img.shape)
        # 将数据传入模型中进行预测

        out = self.model(rimg)
        out = normalize(out)
        embedding = out.detach().cpu().numpy()

        return embedding








    # # 检查输入的关键点坐标是否符合要求，如果不符合则会抛出异常
    # assert landmark.shape[0]==68 or landmark.shape[0]==5
    # assert landmark.shape[1]==2

    # # 如果输入的关键点坐标是68个，则转换为5个关键点坐标
    # # 5个关键点坐标分别是左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
    # # 根据输入的关键点坐标计算出5个关键点的坐标
    # if landmark.shape[0]==68:
    #   landmark5 = np.zeros( (5,2), dtype=np.float32 )
    #   landmark5[0] = (landmark[36]+landmark[39])/2
    #   landmark5[1] = (landmark[42]+landmark[45])/2
    #   landmark5[2] = landmark[30]
    #   landmark5[3] = landmark[48]
    #   landmark5[4] = landmark[54]
    # else:
    #   landmark5 = landmark



    # # 将输入图像进行对齐，以便后续进行特征提取
    # # 使用这5个关键点的坐标进行仿射变换，将输入图像进行对齐
    # tform = trans.SimilarityTransform()  # 存储放射变换矩阵
    # tform.estimate(landmark5, self.src)  # 将输入的5个关键点坐标landmark5和预定义的5个关键点坐标self.src作为参数传递给该方法，以估计仿射变换矩阵
    # M = tform.params[0:2,:]

    # # cv2.warpAffine函数用于进行仿射变换，将其进行对齐
    # img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)





    # # 进行RGB颜色空间的转换，并对图像进行翻转和变换。最后将变换后的图像输入到模型中，得到特征向量feat，将其展平并返回
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # np.fliplr函数用于进行图像翻转
    # img_flip = np.fliplr(img)
    # #img = np.transpose(img, (2,0,1)) #3*112*112, RGB
    # #img_flip = np.transpose(img_flip,(2,0,1))
    # #print(img.shape, img_flip.shape)

    # # self.transform函数用于进行图像变换
    # sample = self.transform(img)
    # sample_flip = self.transform(img_flip)
    # #print(sample.shape, sample_flip.shape)

    # # torch.stack函数用于将变换后的图像打包成一个张量
    # sample_input = torch.stack([sample,sample_flip],0).to(self.device)
    # #print(sample_input.shape)
    # with torch.no_grad():
    #     # self.model.forward函数用于将张量输入到模型中进行特征提取
    #     feat = self.model.forward(sample_input)[0].cpu().numpy()


    # #print(feat.shape)
    # # feat.flatten()函数用于将特征向量展平
    # feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])

    # feat = feat.flatten()
    # return feat
























# class Embedding:
#   # 初始化人脸识别模型
#   # ctx_id表示 GPU设备编号
#   def __init__(self, backbone, prefix, epoch, ctx_id=0):

#       input_size = [112,112]
#       self.device = torch.device("cuda:{}".format(ctx_id))

#       self.model = get_model(backbone, fp16=False).to(self.device)

#       state_dict = torch.load(prefix, map_location=self.device)
#       self.model.load_state_dict(state_dict)
#       self.model.eval()

#       # 用于图像变换的参数
#       RGB_MEAN = [0.5, 0.5, 0.5]
#       RGB_STD = [0.5, 0.5, 0.5]
#       # 图像变换transform
#       self.transform = transforms.Compose([
#            transforms.ToPILImage(),
#            transforms.ToTensor(), 
#            transforms.Normalize(mean = RGB_MEAN,
#                             std = RGB_STD),])
      
#       # 定义src数组和图像大小
#       # src数组是人脸对齐的关键点坐标
#       src = np.array([
#         [30.2946, 51.6963],
#         [65.5318, 51.5014],
#         [48.0252, 71.7366],
#         [33.5493, 92.3655],
#         [62.7299, 92.2041] ], dtype=np.float32 )
#       src[:,0] += 8.0
#       self.src = src
#       self.image_size = input_size


#   # 获取图像的特征向量
#   # 将输入的图像进行对齐、变换和特征提取，并返回特征向量
#   # rimg表示输入的图像，landmark表示输入图像中人脸的关键点坐标
#   def get(self, rimg, landmark):








#     # 检查输入的关键点坐标是否符合要求，如果不符合则会抛出异常
#     assert landmark.shape[0]==68 or landmark.shape[0]==5
#     assert landmark.shape[1]==2

#     # 如果输入的关键点坐标是68个，则转换为5个关键点坐标
#     # 5个关键点坐标分别是左眼中心、右眼中心、鼻尖、左嘴角、右嘴角
#     # 根据输入的关键点坐标计算出5个关键点的坐标
#     if landmark.shape[0]==68:
#       landmark5 = np.zeros( (5,2), dtype=np.float32 )
#       landmark5[0] = (landmark[36]+landmark[39])/2
#       landmark5[1] = (landmark[42]+landmark[45])/2
#       landmark5[2] = landmark[30]
#       landmark5[3] = landmark[48]
#       landmark5[4] = landmark[54]
#     else:
#       landmark5 = landmark



#     # 将输入图像进行对齐，以便后续进行特征提取
#     # 使用这5个关键点的坐标进行仿射变换，将输入图像进行对齐
#     tform = trans.SimilarityTransform()  # 存储放射变换矩阵
#     tform.estimate(landmark5, self.src)  # 将输入的5个关键点坐标landmark5和预定义的5个关键点坐标self.src作为参数传递给该方法，以估计仿射变换矩阵
#     M = tform.params[0:2,:]

#     # cv2.warpAffine函数用于进行仿射变换，将其进行对齐
#     img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)





#     # 进行RGB颜色空间的转换，并对图像进行翻转和变换。最后将变换后的图像输入到模型中，得到特征向量feat，将其展平并返回
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # np.fliplr函数用于进行图像翻转
#     img_flip = np.fliplr(img)
#     #img = np.transpose(img, (2,0,1)) #3*112*112, RGB
#     #img_flip = np.transpose(img_flip,(2,0,1))
#     #print(img.shape, img_flip.shape)

#     # self.transform函数用于进行图像变换
#     sample = self.transform(img)
#     sample_flip = self.transform(img_flip)
#     #print(sample.shape, sample_flip.shape)

#     # torch.stack函数用于将变换后的图像打包成一个张量
#     sample_input = torch.stack([sample,sample_flip],0).to(self.device)
#     #print(sample_input.shape)
#     with torch.no_grad():
#         # self.model.forward函数用于将张量输入到模型中进行特征提取
#         feat = self.model.forward(sample_input)[0].cpu().numpy()


#     #print(feat.shape)
#     # feat.flatten()函数用于将特征向量展平
#     feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])

#     feat = feat.flatten()
#     return feat






































# class Embedding:
#   def __init__(self, prefix, epoch, ctx_id=0):
#       input_size = [112,112]
#       self.device = torch.device("cuda:{}".format(ctx_id))


#       self.model = IR_50(input_size).to(self.device)


#       state_dict = torch.load(prefix, map_location=self.device)
#       self.model.load_state_dict(state_dict)
#       self.model.eval()
#       RGB_MEAN = [0.5, 0.5, 0.5]
#       RGB_STD = [0.5, 0.5, 0.5]

#       self.transform = transforms.Compose([
#            transforms.ToPILImage(),
#            transforms.ToTensor(), 
#            transforms.Normalize(mean = RGB_MEAN,
#                             std = RGB_STD),])
#       src = np.array([
#         [30.2946, 51.6963],
#         [65.5318, 51.5014],
#         [48.0252, 71.7366],
#         [33.5493, 92.3655],
#         [62.7299, 92.2041] ], dtype=np.float32 )
#       src[:,0] += 8.0
#       self.src = src
#       self.image_size = input_size

#   def get(self, rimg, landmark):
#     assert landmark.shape[0]==68 or landmark.shape[0]==5
#     assert landmark.shape[1]==2
#     if landmark.shape[0]==68:
#       landmark5 = np.zeros( (5,2), dtype=np.float32 )
#       landmark5[0] = (landmark[36]+landmark[39])/2
#       landmark5[1] = (landmark[42]+landmark[45])/2
#       landmark5[2] = landmark[30]
#       landmark5[3] = landmark[48]
#       landmark5[4] = landmark[54]
#     else:
#       landmark5 = landmark
#     tform = trans.SimilarityTransform()
#     tform.estimate(landmark5, self.src)
#     M = tform.params[0:2,:]
#     img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_flip = np.fliplr(img)
#     #img = np.transpose(img, (2,0,1)) #3*112*112, RGB
#     #img_flip = np.transpose(img_flip,(2,0,1))
#     #print(img.shape, img_flip.shape)
#     sample = self.transform(img)
#     sample_flip = self.transform(img_flip)
#     #print(sample.shape, sample_flip.shape)
#     sample_input = torch.stack([sample,sample_flip],0).to(self.device)
#     #print(sample_input.shape)
#     with torch.no_grad():
#         feat = self.model.forward(sample_input)[0].cpu().numpy()
#     #print(feat.shape)
#     feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])

#     feat = feat.flatten()
#     return feat

