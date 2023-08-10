
import math
from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize


# 在分布式环境中进行人脸识别
class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    When sample rate less than 1, in each iteration, positive class centers and a random subset of
    negative class centers are selected to compute the margin-based softmax loss, all class
    centers are still maintained throughout the whole training process, but only a subset is
    selected and updated in each iteration.
    .. note::
        When sample rate equal to 1, Partial FC is equal to model parallelism(default sample rate is 1).
    Example:
    --------
    >>> module_pfc = PartialFC(embedding_size=512, num_classes=8000000, sample_rate=0.2)
    >>> for img, labels in data_loader:
    >>>     embeddings = net(img)
    >>>     loss = module_pfc(embeddings, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    """

    # FC 层 的分布式稀疏更新变体，称为 partial FC (PFC)。
    # 当样本率小于 1 时，在每次迭代中，选择正类中心和负类中心的随机子集来计算基于边际的softmax损失，
    # 在整个训练过程中仍然保留所有类中心，但只有一个子集在每次迭代中选择和更新。

    # .. 笔记：：
    #     当采样率等于1时，Partial FC等于模型并行度（默认采样率为1）。
    # 例子：





    _version = 2

    # PartialFC_V2类的 构造函数
    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Paramenters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()

        # 使用assert语句检查 是否 已初始化分布式环境
        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()

        # 初始化一些变量，包括embedding_size、sample_rate、fp16、num_local、class_start、num_sample 和 last_batch_size
        # 这些变量用于计算每个进程负责的类别数量、每个进程负责的类别起始编号、每个进程负责的样本数量等
        self.embedding_size = embedding_size
        self.sample_rate: float = sample_rate
        self.fp16 = fp16
        self.num_local: int = num_classes // self.world_size + int(
            self.rank < num_classes % self.world_size
        )
        self.class_start: int = num_classes // self.world_size * self.rank + min(
            self.rank, num_classes % self.world_size
        )
        self.num_sample: int = int(self.sample_rate * self.num_local)
        self.last_batch_size: int = 0



        self.is_updated: bool = True
        self.init_weight_update: bool = True

        # 初始化一些权重变量，包括 weight。weight是一个PyTorch参数，它是一个形状为(num_local, embedding_size)的张量
        # 其中 num_local 是每个进程负责的类别数量，embedding_size是嵌入向量的维度
        # 权重变量的初始化使用了torch.normal函数，它生成一个 均值为0、标准差为0.01 的正态分布随机数张量
        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (self.num_local, embedding_size)))

        # margin_loss
        # 如果margin_loss是一个可调用对象，则将其赋值给self.margin_softmax
        # 否则，将引发异常
        # margin_loss是一个用于计算 softmax损失 的函数，它可以是 PyTorch 的 CrossEntropyLoss函数 或 其他自定义函数
        if isinstance(margin_loss, Callable):
            self.margin_softmax = margin_loss
        else:
            raise






    # 对标签进行采样，以减少计算量
    def sample(self, labels, index_positive):
        # 接受三个参数，包括 labels、index_positive 和 optimizer
        # labels是一个张量，包含了每个样本的标签
        # index_positive是一个张量，包含了每个正样本的索引
        # optimizer是一个PyTorch优化器，用于更新权重
        """
            This functions will change the value of labels
            Parameters:
            -----------
            labels: torch.Tensor
                pass
            index_positive: torch.Tensor
                pass
            optimizer: torch.optim.Optimizer
                pass
        """
        with torch.no_grad():
            # 首先使用 torch.unique函数 获取正样本的标签
            positive = torch.unique(labels[index_positive], sorted=True).cuda()
            if self.num_sample - positive.size(0) >= 0:
                # 根据 需要采样的 样本数量，生成一个随机排列，并将正样本的索引设置为 2.0
                perm = torch.rand(size=[self.num_local]).cuda()
                perm[positive] = 2.0
                # 使用torch.topk函数 获取排列中 前num_sample个 最大值的索引，并将其排序
                index = torch.topk(perm, k=self.num_sample)[1].cuda()
                # 如果需要采样的样本数量小于正样本的数量，则直接使用正样本的索引
                index = index.sort()[0].cuda()
            else:
                index = positive
            self.weight_index = index
            # 最后，将 labels中 的标签映射到采样后的索引，并返回权重张量的子集
            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        return self.weight[self.weight_index]




    # 前向传播
    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        # 两个参数
        # local_embeddings是一个张量，包含了每个GPU（Rank）上的特征嵌入
        # local_labels是一个张量，包含了每个GPU（Rank）上的标签
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        # 首先将 local_labels 的维度从 2维 降到 1维，并将其转换为 长整型
        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: {self.last_batch_size} vs {batch_size}")

        _gather_embeddings = [
            torch.zeros((batch_size, self.embedding_size)).cuda()
            for _ in range(self.world_size)
        ]
        _gather_labels = [
            torch.zeros(batch_size).long().cuda() for _ in range(self.world_size)
        ]

        # 使用 AllGather函数 将所有GPU（Rank）上的特征嵌入和标签 收集到一个张量中
        _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
        distributed.all_gather(_gather_labels, local_labels)

        embeddings = torch.cat(_list_embeddings)
        labels = torch.cat(_gather_labels)

        labels = labels.view(-1, 1)

        # 将标签映射到 每个进程 负责的类别编号，并将不属于任何类别的标签设置为 -1
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        # 如果采样率小于1，则使用 sample函数 对权重进行采样
        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive)
        else:
            # 否则，直接使用权重
            weight = self.weight

        with torch.cuda.amp.autocast(self.fp16):
            # 使用normalize函数对特征嵌入和权重进行归一化，并使用 linear函数 计算它们之间的线性变换
            norm_embeddings = normalize(embeddings)
            norm_weight_activated = normalize(weight)
            logits = linear(norm_embeddings, norm_weight_activated)
        if self.fp16:
            logits = logits.float()
        logits = logits.clamp(-1, 1)

        # 使用 margin_softmax函数 计算 softmax损失
        logits = self.margin_softmax(logits, labels)
        # 使用dist_cross_entropy函数计算交叉熵损失
        loss = self.dist_cross_entropy(logits, labels)
        return loss








# 自动梯度函数（求导）
# 在分布式训练环境中计算交叉熵损失，其中数据被分割到多个GPU上

# DistCrossEntropyFunc是一个自定义的PyTorch自动求导函数，用于在分布式训练环境中计算交叉熵损失
# 该函数接受两个输入张量：logits和label。它首先计算批次中每个示例的最大分数，然后从所有分数中减去该值以确保数值稳定性
# 接下来，该函数对分数进行指数化，并计算每个示例的 指数化分数之和
# 然后 将分数除以分数之和 以获得 每个类别的softmax概率
# 最后，该函数仅选择具有 有效标签（即不等于-1） 的示例，并为这些示例计算 交叉熵损失
# 该损失使用 distributed.all_reduce方法 在所有GPU上进行了减少操作
class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    # CrossEntropy损失 是并行计算的，将分母全部减少到 单个gpu 中并计算softmax

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        # 接受两个输入张量：logits 和 label
        # logits张量 包含每个类别的预测分数，而 label张量包含批次中每个示例的真实标签
        """ """

        # 首先计算批次中每个示例的最大分数，然后从所有分数中减去该值以确保数值稳定性
        batch_size = logits.size(0)
        # for numerical stability

        # max_logits张量然后
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        # 使用 distributed.all_reduce方法 在所有GPU上 进行减少操作，该方法执行逐元素减少操作，并将结果存储在单个GPU上
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)

        # 对分数进行指数化，并计算每个示例的指数化分数之和
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        # 这些值也使用distributed.all_reduce方法 在所有GPU上 进行了减少操作
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        # 将分数除以分数之和以获得每个类别 的 softmax概率
        logits.div_(sum_logits_exp)
        # 仅选择具有有效标签（即不等于-1）的示例，并为这些示例计算交叉熵损失
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        # 该损失使用distributed.all_reduce方法在所有GPU上进行了减少操作
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        # 对损失应用 对数 和 负号，并返回 结果张量的平均值
        return loss.clamp_min_(1e-30).log_().mean() * (-1)
        # ctx.save_for_backward方法用于保存index、logits和label张量，以便在反向传播中使用
    




    @staticmethod
    # 在自动求导过程 的反向传播中 计算 损失相对于 输入张量的梯度
    def backward(ctx, loss_gradient):
        # 两个参数
        # ctx参数是 在前向传递期间保存的上下文对象，其中包含 index、logits和label 张量
        # loss_gradient参数 是损失 相对于 DistCrossEntropyFunc函数 的输出的梯度

        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        # 首先从上下文对象中解包保存的张量
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        # 从logits张量的大小 计算 批次大小，并创建 与logits张量大小 相同的 独热张量
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        # 使用 scatter_ 方法，在label张量指定的位置上将独热张量填充为1
        one_hot.scatter_(1, label[index], 1)
        # 从 logits张量中 在index张量指定的位置上 减去 独热张量
        logits[index] -= one_hot
        # 将logits张量除以批次大小
        logits.div_(batch_size)
        # 返回损失相对于 logits张量 的梯度，乘以 loss_gradient参数
        # 对于相对于 label张量 的梯度，返回None，因为 label张量 不是 DistCrossEntropyFunc函数的输入
        return logits * loss_gradient.item(), None









# 在分布式训练环境中计算交叉熵损失
class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        # 使用 DistCrossEntropyFunc函数 执行实际计算
        return DistCrossEntropyFunc.apply(logit_part, label_part)






# 在分布式训练环境中执行 all_gather操作，并支持反向传播
# 通过将该函数作为PyTorch自动求导函数实现，用户可以在模型中使用该函数，并在 反向传播 期间自动计算梯度
class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""
    # 接受一个输入张量 和一个 可变数量的张量列表

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        # 使用 distributed.all_gather函数 在所有GPU上执行 all_gather操作
        # 将结果存储在张量列表中
        distributed.all_gather(gather_list, tensor)
        # 返回张量列表作为元组
        return tuple(gather_list)

    # 在反向传递期间，该函数将梯度分配给输入张量，并将梯度 累加到 张量列表中的所有张量中
    @staticmethod
    def backward(ctx, *grads):
        # 首先从 梯度元组 中提取梯度列表，并将 输入张量的梯度 分配给 当前GPU的梯度
        grad_list = list(grads)
        rank = distributed.get_rank()
        grad_out = grad_list[rank]

        dist_ops = [
            # 使用distributed.reduce方法在所有GPU上执行 reduce操作
            # 将梯度累加到张量列表中的所有张量中
            distributed.reduce(grad_out, rank, distributed.ReduceOp.SUM, async_op=True)
            if i == rank
            else distributed.reduce(
               
                grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
            )
            for i in range(distributed.get_world_size())
        ]
        for _op in dist_ops:
            _op.wait()
        # 将输入张量的梯度乘以梯度列表的长度，并返回一个元组，其中第一个元素是输入张量的梯度，其余元素为None
        grad_out *= len(grad_list)  # cooperate with distributed loss function
        return (grad_out, *[None for _ in range(len(grad_list))])



# 变量，通过调用 AllGatherFunc.apply 实现
# 用户可以使用AllGather变量来调用AllGatherFunc函数
AllGather = AllGatherFunc.apply
