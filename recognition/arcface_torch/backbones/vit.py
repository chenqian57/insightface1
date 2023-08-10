# 基于Transformer的视觉模型
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VITBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        return self.bn(x)


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        with torch.cuda.amp.autocast(True):
            batch_size, num_token, embed_dim = x.shape
            #qkv is [3,batch_size,num_heads,num_token, embed_dim//num_heads]
            qkv = self.qkv(x).reshape(
                batch_size, num_token, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 num_patches: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Callable = nn.ReLU6,
                 norm_layer: str = "ln", 
                 patch_n: int = 144):
        super().__init__()

        if norm_layer == "bn":
            self.norm1 = VITBatchNorm(num_features=num_patches)
            self.norm2 = VITBatchNorm(num_features=num_patches)
        elif norm_layer == "ln":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.extra_gflops = (num_heads * patch_n * (dim//num_heads)*patch_n * 2) / (1000**3)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class PatchEmbed(nn.Module):
    def __init__(self, img_size=108, patch_size=9, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x







class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    # img_size：输入图像的大小。       patch_size：每个图像块的大小。         in_channels：输入图像的通道数。
    # num_classes：分类器的输出维度。  embed_dim：嵌入向量的维度。            depth：变换器的深度。
    # num_heads：多头注意力机制中头的数量。 mlp_ratio：MLP 层中隐藏层大小与嵌入向量大小之比。 qkv_bias：是否在注意力机制中使用偏置。
    # qk_scale：注意力机制中的缩放因子。drop_rate：输入和输出的 dropout 概率。 attn_drop_rate：注意力机制中的 dropout 概率。
    # drop_path_rate：随机删除路径的概率。  hybrid_backbone：混合 CNN 输入阶段的模型。  norm_layer：归一化层的类型。
    # mask_ratio：用于遮蔽的比率。     using_checkpoint：是否使用检查点。
    def __init__(self,
                 img_size: int = 112,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,

                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: Optional[None] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,

                 drop_path_rate: float = 0.,
                 hybrid_backbone: Optional[None] = None,
                 norm_layer: str = "ln",
                 mask_ratio = 0.1,
                 using_checkpoint = False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        # 保持与其他模型的一致性
        self.num_features = self.embed_dim = embed_dim


        # 检查 hybrid_backbone 参数是否为 None。如果不是，则引发 ValueError 异常
        if hybrid_backbone is not None:
            raise ValueError
        # 否则，它使用 PatchEmbed 类创建一个 patch_embed 对象，该对象将输入图像分成图像块，并将每个图像块转换为嵌入向量
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        # 将 mask_ratio 和 using_checkpoint 参数分别赋值给  self.mask_ratio 和 self.using_checkpoint
        self.mask_ratio = mask_ratio
        self.using_checkpoint = using_checkpoint
        num_patches = self.patch_embed.num_patches

        # 计算图像块的数量，并将其赋值给 self.num_patches
        self.num_patches = num_patches
        # 使用 nn.Parameter 创建一个形状为 (1, num_patches, embed_dim) 的张量，并将其赋值给 self.pos_embed(位置编码矩阵，用于将每个块的位置信息编码到特征向量中)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # 使用 nn.Dropout 创建一个 pos_drop 对象，该对象将在嵌入向量中应用 dropout
        self.pos_drop = nn.Dropout(p=drop_rate)



        # 创建 VisionTransformer 的多个块
        # stochastic depth decay rule 随机深度衰减规律
        # 使用 torch.linspace 函数生成一个长度为 depth 的等差数列，然后将其转换为列表 dpr
        # dpr 中的每个元素都是一个 浮点数，表示在每个块中应用 dropout 的概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # 计算图像块的数量，并将其赋值给 patch_n
        patch_n = (img_size//patch_size)**2
        # 使用 nn.ModuleList 创建一个模块列表 self.blocks，其中每个模块都是 Block 类的一个实例
        # Block 类是一个基本的变换器块，用于将嵌入向量进行自注意力计算和 MLP 计算
        # 每个块都接受多个参数，包括 dim、num_heads、mlp_ratio、qkv_bias、qk_scale、drop、attn_drop、drop_path、norm_layer、num_patches 和 patch_n。
        
        # dim：嵌入向量的维度。                  num_heads：多头注意力机制中头的数量。 mlp_ratio：MLP 层中隐藏层大小与嵌入向量大小之比。
        # qkv_bias：是否在注意力机制中使用偏置。 qk_scale：注意力机制中的缩放因子。    drop：输入和输出的 dropout 概率。
        # attn_drop：注意力机制中的 dropout 概率。drop_path：随机删除路径的概率。      norm_layer：归一化层的类型。
        # num_patches：图像块的数量。            patch_n：每个图像块的大小
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                      num_patches=num_patches, patch_n=patch_n)
                for i in range(depth)]
        )
        
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops
        # 使用 nn.LayerNorm 或 VITBatchNorm 创建一个归一化层 self.norm，具体取决于 norm_layer 参数的值
        # 如果 norm_layer 为 "ln"，则使用 nn.LayerNorm 创建一个层归一化层
        # 如果 norm_layer 为 "bn"，则使用 VITBatchNorm 创建一个批归一化层
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(embed_dim)
        elif norm_layer == "bn":
            self.norm = VITBatchNorm(self.num_patches)




        # features head
        # 特征提取器
        # nn.Sequential对象 包含四个层：包含2个线性层、2个批归一化层
        # nn.Linear 是一个线性层，可以将输入特征映射到输出特征。nn.BatchNorm1d 是一个批归一化层，可以对输出进行归一化
        # emded_dim表示特征的维度；num_patches表示图像块的数量；num_classes表示分类器的输出维度
        self.feature = nn.Sequential(
            # 第一个线性层将输入的特征展平为一维张量，然后将其输入到一个具有 embed_dim 个输出特征的线性层中
            nn.Linear(in_features=embed_dim * num_patches, out_features=embed_dim, bias=False),
            # 第一个批归一化对输出进行归一化
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            # 第二个线性层将输入的特征映射到一个具有 num_classes 个输出特征的空间中
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            # 第二个批归一化对输出进行归一化
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5)
        )


        # 初始化 vit 模型参数，mask_token参数形状为(1, 1, embed_dim),用于遮蔽输入序列中的特殊标记
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 使用正态分布随机初始化，均值为0，标准差为0.02
        torch.nn.init.normal_(self.mask_token, std=.02)
        # pos_embed 参数，表示位置嵌入矩阵，用于将输入序列中的位置信息编码到特征向量中
        # 使用截断正态分布初始化 pos_embed 参数，均值为0，标准差为0.02
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        # 调用_init_weights函数，对模型参数进行初始化
        self.apply(self._init_weights)



    # 初始化vit模型的权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_函数，使用截断正态分布初始化张量。初始化线性层的权重
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # nn.init.constant_, 使用常数初始化张量。
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # 初始化偏置项和 批归一化层的权重和偏置项
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    



    # patch_embed 函数将输入的图像分割为多个块，并将每个块转换为一个特征向量
    # pos_embed 是一个位置编码矩阵，用于将每个块的位置信息编码到特征向量中
    # pos_drop 函数是一个随机丢弃层，可以随机丢弃位置编码矩阵中的一些元素，以增加模型的鲁棒性和泛化能力
    # random_masking 函数可以用于在训练过程中对输入序列进行随机掩盖，以增加模型的鲁棒性和泛化能力
    # blocks 是一个由多个 Block 对象组成的列表，每个 Block 对象包含了多个 Attention 和 Mlp 层，用于对输入特征进行处理
    # mask_token 是一个超参数，表示用于掩蔽输入序列中的特殊标记
    # mask_ratio 是一个超参数，表示掩盖比例


    # 这个函数可以用于在训练过程中对输入序列进行随机掩盖，以增加模型的鲁棒性和泛化能力
    # 对输入的序列进行随机掩蔽
    def random_masking(self, x, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # 首先从 输入序列 的 最后一个维度中获取序列的长度L 和特征维度D
        # N表示输入序列的批次大小，L表示输入序列的长度，D表示输入序列的特征维度
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # 生成一个形状为(N, L)的随机噪声张量，噪声范围为[0, 1]，并将其赋值给 noise
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        # 使用torch.argsort函数 对噪声进行排序，升序排列，小的保留，大的删除。生成一个排序后的索引张量ids_shuffle
        # 以便对每个样本进行随机掩蔽
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # 使用 ids_shuffle 对输入序列进行索引，以生成一个掩蔽后的序列x_masked
        ids_keep = ids_shuffle[:, :len_keep]

        # torch.gather 函数根据索引张量 ids_keep 和 ids_restore 对输入序列进行索引，以生成掩盖后的序列 x_masked 和恢复原始顺序的掩码张量 mask
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # 最后，代码生成一个二进制掩码张量mask，其中0表示保留，1表示掩盖
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        # 掩码张量mask 与排序后的索引张量ids_restore 进行索引，以生成一个恢复原始顺序的掩码张量
        mask = torch.gather(mask, dim=1, index=ids_restore)


        return x_masked, mask, ids_restore











    # 用于前向传播输入数据并生成特征向量
    def forward_features(self, x):
        B = x.shape[0]
        # 将输入的图像x 通过 patch_embed 函数转换为一个形状为(N, num_patches, embed_dim)的张量
        # 其中N是输入图像的批次大小，num_patches是图像分割后的块数，embed-dim是每个块的特征维度
        x = self.patch_embed(x)
        # 使用pos_embed对每个块进行位置编码
        x = x + self.pos_embed
        # 对位置编码后的块进行dropout
        x = self.pos_drop(x)

        if self.training and self.mask_ratio > 0:
            # 使用random_masking函数对输入序列进行随机掩盖
            x, _, ids_restore = self.random_masking(x)

        # 使用blocks中的每个块对块进行特征提取
        for func in self.blocks:
            if self.using_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(func, x)
            else:
                x = func(x)
        x = self.norm(x.float())
        
        if self.training and self.mask_ratio > 0:
            # 使用mask_token对掩盖的块进行填充
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = x_
        # 将特征向量 reshape 为(N, num_patches * embed_dim)的张量，并返回该张量
        return torch.reshape(x, (B, self.num_patches * self.embed_dim))






    def forward(self, x):
        # 使用forward_features函数生成特征向量
        x = self.forward_features(x)
        # 使用feature函数对特征向量进行分类
        x = self.feature(x)
        return x


