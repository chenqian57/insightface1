from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)



    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)



    # Vision Transformer（ViT）
    # drop_path_rate是随机删除路径的概率，用于正则化模型
    # 在vit_t中，drop_path_rate默认为0.1，而在vit_t_dp005_mask0中，drop_path_rate默认为0.05
    # mask_ratio是用于遮蔽的比率，用于在训练期间随机遮蔽输入图像的一部分，以增强模型的鲁棒性
    # 在 vit_t中，mask_ratio 默认为0.1，而在 vit_t_dp005_mask0 中，mask_ratio默认为0.0，即不使用遮蔽
    # vit_t可能会更加鲁棒，因为它使用了遮蔽和更高的drop_path_rate，但这也可能会导致更多的过拟合
    # 而vit_t_dp005_mask0可能会更加稳定，因为它没有使用遮蔽，并且使用了更小的drop_path_rate，但这也可能会导致模型欠拟合

    elif name == "vit_t":
        # 获取关键字参数num_features的值，如果没有则返回512
        num_features = kwargs.get("num_features", 512)
        # 从 vit 模块中导入 VisionTransformer 类，并使用该类创建一个 VisionTransformer 对象
        from .vit import VisionTransformer
        return VisionTransformer(
            # img_size：输入图像的大小。           patch_size：每个图像块的大小。 num_classes：分类器的输出维度。
            # embed_dim：嵌入向量的维度。          depth：Transformer的深度。   num_heads：多头注意力机制中头的数量。
            # drop_path_rate：随机删除路径的概率。 norm_layer：归一化层的类型。  mask_ratio：用于遮蔽的比率
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    # 
    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)



    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)


    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    

    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)


    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)


    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)


    else:
        raise ValueError()
