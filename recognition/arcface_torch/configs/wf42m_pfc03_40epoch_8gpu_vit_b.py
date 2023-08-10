from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp  # 在/train_tmp目录下挂载大小为140GB的tmpfs文件系统。tmpfs文件系统是一个临时文件存储系统

config = edict()  # 点表示法
config.margin_list = (1.0, 0.0, 0.4)  # 
config.network = "vit_b_dp005_mask_005"  # vit_b模型，使用Mask RCNN在COCO上进行预训练
config.resume = False  # 控制训练过程中是否从之前的检查点恢复训练
config.output = None
config.embedding_size = 512
config.sample_rate = 0.3  # 采样率
config.fp16 = True  # 半精度浮点数
config.weight_decay = 0.1  # 正则化，向损失函数添加一个惩罚项，以减少权重的大小
config.batch_size = 256
config.gradient_acc = 12 # total batchsize is 256 * 12，gradient_acc是梯度累积的数量
config.optimizer = "adamw"  # Adam优化器的一种变体
config.lr = 0.001
config.verbose = 2000  # 在测试期间打印更多消息
config.dali = False

config.rec = "/train_tmp/WebFace42M"  # 数据集路径
config.num_classes = 2059906  # 分类个数
config.num_image = 42474557   # 训练集中图片数量
config.num_epoch = 40  # 训练的总epoch数
config.warmup_epoch = config.num_epoch // 10  # 取整数。训练开始时，学习率从较小的值开始逐渐增大到一个较大的值的过程所占用的epoch数
config.val_targets = []  # 
