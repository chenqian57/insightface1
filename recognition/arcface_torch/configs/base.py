from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"   # r50,
config.save_all_states = False
config.output = "ms1mv3_arcface_r50"
# ms1mv3_arcface_r50

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False

# 128, 100
config.batch_size = 100

# For SGD
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10



# config.rec = "/home/qiujing/face_data/webface26m/0/zong_0"    # 指定训练数据的存储路径
# # /home/qiujing/face_data/webface26m/0/zong_0
# config.num_classes = 411980
# config.num_image = 8465386








# For Large Sacle Dataset, such as WebFace42M
config.dali = False 

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# WandB Logger
config.wandb_key = "00d76c4fa6f799354037bf6fc124558afb33db14"
# 00d76c4fa6f799354037bf6fc124558afb33db14
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
config.suffix_run_name = None
config.using_wandb = True
# config.wandb_entity = "entity"
# config.wandb_project = "project"
config.wandb_entity = "chenq7"
config.wandb_project = "insightface1"
config.wandb_log_all = True
config.save_artifacts = True
config.wandb_resume = True # resume wandb run: Only if the you wand t resume the last run that it was interrupted
                            # resume wandb run:仅当您不想恢复上次被中断的运行时

config.notes = "Face Recognition"


