import importlib   # 使用 Python 的 importlib 模块动态导入配置文件，并将其与默认配置信息进行合并
import os.path as osp


# 为深度学习模型提供配置信息，例如模型的超参数、数据集路径、输出路径等
def get_config(config_file):
    # 检查 config_file 是否以 configs/ 开头，如果不是则会抛出异常
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    # configs/
    # /home/qiujing/cqwork/insightface/recognition/arcface_torch/configs/


    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    # 从 configs.base 模块中导入 config 变量，该变量包含了默认的配置信息
    # config = importlib.import_module(".configs.glint360k_r50", package='home.qiujing.cqwork.insightface.recognition.arcface_torch')
    config = importlib.import_module(".wf2601_r50", package='configs')

    # .wf2601_r50
    # .glint360k_r50

    # configs.glint360k_r50
    # configs.base
    # home.qiujing.cqwork.insightface.recognition.arcface_torch.configs.glint360k_r50

    # 它从指定的配置文件中导入 config 变量，并将其与默认配置信息进行合并
    cfg = config.config

    # 最后，它检查配置信息中是否指定了输出路径，如果没有则将输出路径设置为 work_dirs/temp_module_name，
    # 其中 temp_module_name 是配置文件的文件名（不包含扩展名）
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('/mnt/ssd/qiujing/arcface/insightface_logs', 'wf26m01_vit_2')
        # cfg.output = osp.join('work_dirs', temp_module_name)
        # cfg.output = osp.join('/mnt/ssd/qiujing/arcface/insightface_logs', temp_module_name)
        # cfg.output = osp.join('/mnt/ssd/qiujing/arcface/insightface_logs', 'wf2601_r50_2')
        # cfg.output = osp.join('/mnt/ssd/qiujing/arcface/insightface_logs', 'wf26m01_r50_3')
        # cfg.output = osp.join('/mnt/ssd/qiujing/arcface/insightface_logs', 'wf26m01_vit_1')
        



    return cfg
