from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
# config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
# True、False
config.momentum = 0.9
config.weight_decay = 1e-4

# 128, 64, 100
config.batch_size = 100

config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/mnt/ssd/qiujing/glint360k"    # 指定训练数据的存储路径
# /train_tmp/glint360k
# /mnt/ssd/qiujing/glint360k
config.num_classes = 360232
config.num_image = 17091657



# config.rec = "/home/qiujing/face_data/webface26m/0/0_0/WebFace260M"    # 指定训练数据的存储路径
# config.num_classes = 30000
# config.num_image = 608184



config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]





# Margin Base Softmax
config.save_all_states = False
config.interclass_filtering_threshold = 0

# For SGD 
config.optimizer = "sgd"
# config.weight_decay = 5e-4


# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.frequent = 10

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
# 2, 44
config.num_workers = 2

# WandB Logger
config.wandb_key = "00d76c4fa6f799354037bf6fc124558afb33db14"
# 00d76c4fa6f799354037bf6fc124558afb33db14
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted
                            # resume wandb run:仅当您不想恢复上次被中断的运行时



