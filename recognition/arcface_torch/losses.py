import torch
import math


# 组合的边界损失函数
# 接受一个 logits 张量和一个 标签张量 作为输入，并计算损失
# logits 张量是一个包含每个类别得分的张量，标签张量是一个包含每个样本的标签的张量



class CombinedMarginLoss(torch.nn.Module):
    # 5个参数， s、m1、m2、m3 和 interclass_filtering_threshold，用于调整损失函数的行为
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    # 在前向传递过程中，损失函数会根据标签张量选择正样本，并计算损失
    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        # 如果 interclass_filtering_threshold 大于 0，则会进行一些额外的处理，以过滤掉得分低于阈值的样本
        # 最后，损失函数会返回一个损失张量
        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]



        # 如果 self.m1 等于 1.0 且 self.m3 等于 0.0，则对 target_logit 和 logits 张量中的每个元素进行反余弦函数计算，然后将 target_logit 张量中的每个元素加上 self.m2，
        # 并将结果赋值给 logits 张量中的一部分元素
        # 接着对 logits 张量中的每个元素进行余弦函数计算，最后将 logits 张量中的每个元素乘以 self.s
        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        # 如果 self.m3 大于 0，则将 target_logit 张量中的每个元素减去 self.m3，并将结果赋值给 logits 张量中的一部分元素。
        # 最后将 logits 张量中的每个元素乘以 self.s
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits





# ArcFace中是直接在角度空间 θ 中最大化分类界限，而 CosFace 是在余弦空间 cos(θ) 中最大化分类界限
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    # ArcFace 损失函数的核心思想是将样本的特征向量映射到一个高维空间中，并在该空间中进行分类
    # 在该空间中，样本之间的距离可以更好地反映它们之间的相似度，从而提高分类的准确率
    # 为了实现这一目标，ArcFace 损失函数引入了 margin 和 scale 两个超参数，用于调整样本之间的距离
    # 其中，margin 用于控制样本之间的距离，scale 用于控制分类的难度
    # 通过调整这两个超参数，可以获得更好的分类效果
    
    # 参数：s 和 margin
    # 接受一个 logits 张量和一个标签张量作为输入，并计算损失
    # def __init__(self, s=64.0, margin=0.5):
    def __init__(self, s, margin):
        super(ArcFace, self).__init__()
        self.scale = s
        self.margin = margin

        # 根据 margin 计算常量，如余弦值和正弦值
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        # 当样本的余弦相似度小于 margin 时，不进行 margin 的调整
        # 当 easy_margin 为 False 时，表示使用标准的 ArcFace 策略，即对所有样本都进行 margin 的调整
        self.easy_margin = False


    # logits 张量是一个包含每个类别 得分 的张量，
    # 标签张量是一个包含每个样本/home/qiujing/cqwork/3FFC的 标签 的张量
    # 在 前向传递 过程中，损失函数 会根据 标签张量 选择 正样本，并计算损失
    # 最后，返回一个损失张量
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        # 获取正样本的索引(即标签不为 -1 的样本)
        index = torch.where(labels != -1)[0]
        # 根据上述索引提取出对应的 logits 和 标签，保存在 target_logit 中
        target_logit = logits[index, labels[index].view(-1)]
        with torch.no_grad():
            # 计算 target_logits 和所有 logits 的余弦值
            target_logit.arccos_()
            logits.arccos_()
            # 根据 margin 调整 target_logit 的余弦值
            final_target_logit = target_logit + self.margin
            # 将 调整后的 target_logit 的余弦值 替换回原来的logits中
            logits[index, labels[index].view(-1)] = final_target_logit
            # 将所有logits的余弦值转换为角度值
            logits.cos_()

        # 将所有的 logits 乘以 scale参数
        logits = logits * self.s
        return logits














# CosFace(https://arxiv.org/pdf/1801.09414v2.pdf)
class CosFace(torch.nn.Module):
    # 参数，如 s 和 m
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    # 接受一个 logits 张量和一个标签张量作为输入，并计算损失
    # 其中，logits 张量是一个包含每个类别得分的张量
    # 标签张量是一个 包含每个样本的标签 的 张量
    # 这些参数用于调整损失函数的行为

    # 在前向传递过程中，损失函数会根据标签张量选择正样本，并计算损失
    # 最后，损失函数会返回一个损失张量
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit

        logits = logits * self.s
        return logits









import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

# 修改Arcface，参照 arcface-torch
# 将样本的特征向量映射到一个高维空间中，并在该空间中进行分类
class Arcface2(torch.nn.Module):
    # def __init__(self, embedding_size=512, num_classes=411980, s=64.0, m=0.5):
    def __init__(self, embedding_size, num_classes, s, margin):
        super(Arcface2, self).__init__()
        self.scale = s  # scale 参数，用于控制分类的难度
        self.margin = margin  # margin 参数，用于调整样本之间的距离


        # # weight，使用了 Xavie 初始化方法，可以有效地提高模型的训练效果
        # self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        
        # # weight需要放进 网络 训练，因此需要传进GPU里
        # if self.use_gpu:
        #     self.weight = nn.Parameter(torch.randn(num_classes, embedding_size).cuda())
        # else:
        #     self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        
        
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size).cuda())
        nn.init.xavier_uniform_(self.weight)


        # 定义常量，其中，余弦值 和正弦值 可以通过三角函数计算得到
        # 而 cos_m 和 sin_m 则是通过 margin 参数计算得到
        # th 和 mm 则是通过 余弦值和正弦值 计算得到
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin


    def forward(self, input: torch.Tensor, label: torch.Tensor):
    # 2个参数，一个输入张量和一个标签张量。输入张量是一批特征向量，标签张量是每个特征向量的标签
        # 使用函数F.linear，计算 输入张量 和 权重张量 之间的余弦相似度
        # F.linear 使用权重张量对输入张量进行线性变换。然后对结果张量进行归一化，以确保其值在-1和1之间
        cosine  = F.linear(input, F.normalize(self.weight))
        # sqrt 和 clamp函数 计算输入 张量和权重张量之间的角度的正弦值
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi = torch.where(label != -1)[0]
        # 使用 ArcFace公式 将余弦和正弦张量 应用于 phi张量 的计算
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)


        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 为了确保仅对批次中存在的标签进行修改
        # 将 one-hot标签张量 乘以 phi张量 并将其加到 余弦张量 乘以（1-one-hot标签张量）
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output

















