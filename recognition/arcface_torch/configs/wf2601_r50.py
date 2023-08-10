from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
# config.margin_list = (1.0, 0.0, 0.4)
# (1.0, 0.5, 0.0)
# (1, 0.3, 0.2)
config.margin_list = (1.0, 0.5, 0.0)


config.margin=0.5



# "r50", "vit_t_dp005_mask0", "vit_t",
config.network = "vit_t"
config.output = None
config.embedding_size = 512





# 1.0, 0.3
config.sample_rate = 1.0
config.fp16 = False
# True、False
config.momentum = 0.9


# 128, 64, 100, 80, 96
config.batch_size = 96

config.lr = 0.1
config.verbose = 2000
config.dali = False    # 数据增强

config.rec = "/mnt/ssd/qiujing/webface26m_zong01"    # 指定训练数据的存储路径
# /train_tmp/glint360k
# /mnt/ssd/qiujing/glint360k
# /home/qiujing/face_data/webface26m/0/0_0/WebFace260M
# /home/qiujing/face_data/webface26m/0/zong_0，txt文件：/home/qiujing/face_data/webface26m/0/01cls_train/wf26m01_cls_train.txt
# /mnt/ssd/qiujing/webface26m_zong01




# # /home/qiujing/face_data/webface26m/0/0_0/WebFace260M
# config.num_classes = 30000
# config.num_image = 608184


# /home/qiujing/face_data/webface26m/0/zong_0
config.num_classes = 411980
config.num_image = 8465386




# 20, 40, 25
config.num_epoch = 20

# 0, config.num_epoch // 10
config.warmup_epoch = 0




config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]


# Margin Base Softmax
config.interclass_filtering_threshold = 0

# For SGD 
config.optimizer = "sgd"
config.weight_decay = 5e-4





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
# 2, 44, 20, 30
config.num_workers = 30





# WandB Logger
config.wandb_key = "00d76c4fa6f799354037bf6fc124558afb33db14"
# 00d76c4fa6f799354037bf6fc124558afb33db14
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
config.suffix_run_name = None

# False、True
config.using_wandb = False
config.wandb_entity = "chenq7"
config.wandb_project = "insightface1"
config.wandb_log_all = False
# False、True
config.save_artifacts = False

# False, True
config.wandb_resume = False # resume wandb run: Only if the you wand t resume the last run that it was interrupted
                            # resume wandb run:仅当您不想恢复上次被中断的运行时
# config.notes = "Face Recognition"


config.save_all_states = True
# False, True
config.resume = False

