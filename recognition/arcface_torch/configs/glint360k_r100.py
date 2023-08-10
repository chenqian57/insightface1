from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False
# 如果后面需要Fine Tune模型，需要将config.py的config.ckpt_embedding这一行的值改为False，这样可以保存网络的fc7层的权重，否则不会保存，不保存就会从头开始重新训练。 


config.rec = "/train_tmp/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]   # 'cfp_ff', 'cfp_fp'
# 在训练评估时可以同时评估cpf_ff数据集，想看模型在该数据集上的准确率可以加上，该数据集上的准确率与LFW的准确率接近。
