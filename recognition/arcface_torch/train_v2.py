import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss, ArcFace, CosFace
# CombinedMarginLoss，ArcFace，CosFace

from lr_scheduler import PolyScheduler
from partial_fc_v2 import PartialFC_V2


from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."


# 环境变量中没有定义这些变量，则默认使用单进程模式，并使用 distributed.init_process_group() 方法初始化分布式训练的后端为 nccl，
# 并指定初始化方法、总进程数和当前进程的全局排名
# 在初始化完成后，所有进程都可以使用 PyTorch 的分布式 API 进行通信和同步
try:
    rank = int(os.getenv("RANK", -1))
    local_rank = int(os.getenv("LOCAL_RANK", 3))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    # rank = 1            # 1, 0
    # local_rank = 3      # 3, 
    # world_size = 1      # -1
    distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        # init_method="env://",
        # tcp://127.0.0.1:12584
        # tcp://10.10.8.217:12584
        rank=rank,
        world_size=world_size,
    )



# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-c",
#         "--config",
#         type=str,
#         default="/home/qiujing/cqwork/insightface/recognition/arcface_torch/configs/glint360k_r50.yaml",
#         # /home/qiujing/cqwork/insightface/recognition/arcface_torch/configs/glint360k_r50.py
#         help="config file",
#     )
#     opt = parser.parse_args()
#     return opt



def main(args):
    cfg = get_config(args.config)
    # global control random seed
    cfg.seed = int(time.time())
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # 使用 torch.cuda.set_device 函数设置当前进程使用的 GPU 设备
    torch.cuda.set_device(local_rank)

    # 创建一个输出目录，并使用 init_logging 函数初始化日志记录器
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)


    # 它创建一个 TensorBoard 的 SummaryWriter 对象和一个 WandB 的 wandb.init 对象（如果使用 WandB 记录器的话）
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )




    # WandB 记录器是一种用于记录深度学习模型训练过程的工具，可以记录模型的指标、超参数、日志等信息，并可视化展示。
    # 通过使用 WandB 记录器，可以方便地监控模型的训练过程，并进行实验管理和结果复现

    # 如果使用 WandB 记录器，该函数会尝试登录 WandB 并初始化记录器对象
    # 在初始化记录器对象时，它会使用配置文件中指定的 WandB 项目和实体名称，并设置记录器的名称、笔记和其他参数。
    # 如果登录或初始化失败，则会输出错误信息
    wandb_logger = None
    # 首先检查配置文件中是否指定了使用 WandB 记录器。如果指定了，则会尝试登录 WandB，如果登录失败则会输出错误信息
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        # 然后，它使用 当前时间 和 进程编号 创建一个记录器名称，并根据 配置文件 中的设置添加后缀
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        # 它尝试初始化 WandB 记录器对象，并设置记录器的实体名称、项目名称、同步 TensorBoard、恢复、名称、笔记等参数
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name) if rank == 0 or cfg.wandb_log_all else None

                # , 
                # notes = cfg.notes


            # 如果初始化成功，则会返回 WandB 记录器对象，并将配置信息更新到记录器对象中。如果初始化失败，则会输出错误信息
            if wandb_logger:
                # wandb_logger.config.update(cfg)
                wandb_logger.config.update(cfg, allow_val_change=True)


        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")



    # 加载训练数据
    # 包括配置文件中指定的 REC 文件路径、本地进程的排名、批次大小、是否使用 DALI 数据增强、随机种子和工作进程数等
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )


    # 创建神经网络模型
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    # 使用 torch.nn.parallel.DistributedDataParallel 函数 创建一个分布式数据并行模型，并将其移动到 GPU 上
    # 在创建分布式数据并行模型时，它指定了模型、设备 ID、桶容量和是否查找未使用的参数等参数
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)



    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    # margin_loss = ArcFace(
    #     64,
    #     0.5
    # )


    # margin_loss = ArcFace(
    #     64,
    #     cfg.margin
    # )


    # margin_loss = Arcface2(
    #     cfg.embedding_size,
    #     cfg.num_classes,
    #     64,
    #     cfg.margin,
    # )












    # 定义一个优化器和学习率调度器，并初始化它们的参数
    if cfg.optimizer == "sgd":
        # 在函数内部，它首先检查配置文件中指定的优化器类型，并根据不同的类型创建不同的优化器
        # 如果使用 SGD 优化器，则会创建一个 SGD 优化器，并将模型的参数和 PartialFC_V2 的参数分别传递给优化器
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)




    # 如果使用 AdamW 优化器，则会创建一个 AdamW 优化器，并将模型的参数和 PartialFC_V2 的参数分别传递给 优化器
    # 如果未指定优化器类型，则会抛出异常
    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise


    # 计算 总批次大小 和 预热步数，并使用 PolyScheduler 类创建一个学习率调度器
    # PolyScheduler 是一种多项式学习率调度器，可以根据训练步数动态调整学习率
    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )






    # 训练模型，并在训练过程中调用回调函数
    # 在函数内部，它遍历训练数据集，并在每个 epoch 中训练模型
    # 在每个 epoch 中，它遍历训练数据集，并在每个批次中计算模型的损失和梯度，并使用优化器更新模型的参数
    # 在每个 epoch 结束时，它调用学习率调度器更新学习率，并调用回调函数进行验证和日志记录
    # 在训练结束时，它保存模型参数、优化器状态和学习率调度器状态，并关闭日志记录器
    # 0, 6, 8, 18, 19, 21
    
    start_epoch = 0

    global_step = 0



    # 在训练过程中从检查点文件中恢复模型的状态。
    # 具体来说，代码首先使用 torch.load 函数从检查点文件中加载字典对象 dict_checkpoint，该字典对象包含了模型的状态、优化器的状态和学习率调度器的状态等信息。
    # 然后，从 dict_checkpoint 中获取模型的状态字典 state_dict_backbone 和 state_dict_softmax_fc，并使用 load_state_dict 函数将这些状态加载到 backbone.module 和 module_partial_fc 中。
    # 接下来，从 dict_checkpoint 中获取优化器的状态字典 state_optimizer，并使用 load_state_dict 函数将这个状态加载到 opt 中。
    # 最后，从 dict_checkpoint 中获取学习率调度器的状态字典 state_lr_scheduler，并使用 load_state_dict 函数将这个状态加载到 lr_scheduler 中。
    # 加载完毕后，模型的状态、优化器的状态和学习率调度器的状态将与检查点文件中保存的状态完全一致。

    # if cfg.wandb_resume:
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])

        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint







    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))


    # 两个回调函数，分别是 callback_verification 和 callback_logging，用于监控和记录训练过程
    # 用于在每个 epoch 结束时进行验证，并记录验证结果
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )

    # 用于在训练过程中记录日志，并在训练结束时记录训练结果
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )


    # 定义两个变量 loss_am 和 amp
    # loss_am 是一个用于计算平均损失的工具类，它会在训练过程中记录每个批次的损失，并计算它们的平均值
    # amp 是 PyTorch 的自动混合精度工具，它可以加速深度学习模型的训练过程
    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        # 使用一个循环来遍历训练数据集，并在每个批次中计算模型的损失和梯度，并使用优化器更新模型的参数
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            # 首先调用模型的前向传播函数 backbone，将输入图像 img 转换为局部特征向量 local_embeddings
            local_embeddings = backbone(img)
            # 它使用 module_partial_fc 函数计算局部特征向量和标签之间的损失，并将损失存储在 loss 变量中
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            # 它根据配置文件中的设置选择使用自动混合精度训练或普通训练
            # 如果使用自动混合精度训练，则会使用 PyTorch 的 GradScaler 类自动缩放损失的梯度，并在每个 cfg.gradient_acc 步后更新模型的参数
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            
            # 如果使用普通训练，则会直接计算损失的梯度，并在每个 cfg.gradient_acc 步后更新模型的参数
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    # 在更新模型参数之前，它还会使用 torch.nn.utils.clip_grad_norm_ 函数对梯度进行裁剪，以避免梯度爆炸的问题
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            # 在更新模型参数之后，它会调用学习率调度器 lr_scheduler 的 step 函数更新学习率，并使用回调函数记录训练过程中的损失和精度
            # 在每个 epoch 结束时，它会保存模型参数、优化器状态和学习率调度器状态，并关闭日志记录器
            lr_scheduler.step()


            # 记录 训练损失 和 精度 的代码
            # 使用 torch.no_grad() 上下文管理器 关闭梯度计算，并记录当前批次的损失和平均损失 到 wandb_logger 中
            # wandb_logger 是一个用于记录训练过程的工具，它可以将训练过程中的指标记录到 W&B 平台上，以便于后续分析和可视化
            with torch.no_grad():
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
                # 它使用 loss_am 更新平均损失，并使用回调函数 callback_logging 记录训练过程中的损失和精度
                # callback_logging 是一个回调函数，它定义在 train_v2.py 文件中，用于记录训练过程中的损失和精度
                # 在这里，它会记录当前的全局步数 global_step、平均损失 loss_am、当前 epoch epoch、是否使用自动混合精度训练 cfg.fp16、
                # 当前学习率 lr_scheduler.get_last_lr()[0] 和自动混合精度工具 amp 的状态
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # 最后，它会在每个 cfg.verbose 步后调用回调函数 callback_verification 进行验证
                # callback_verification 是一个回调函数，它定义在 train_v2.py 文件中，用于在训练过程中进行验证
                # 在这里，它会在每个 cfg.verbose 步后调用模型的前向传播函数 backbone 进行验证，并记录验证结果
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)




        # 总的来说，这段代码是在训练过程中保存模型参数和优化器状态的代码
        # 它会根据配置文件中的设置选择是否保存所有状态或仅保存模型参数，并使用 wandb_logger 记录模型参数到 W&B 平台上
        # 如果设置了 cfg.dali 为 True，则会在每个 epoch 结束后重置数据加载器

        # 在训练过程中保存模型参数和优化器状态
        # 在每个 epoch 结束后，它会根据配置文件中的设置选择是否保存所有状态或仅保存模型参数
        # 如果设置了 cfg.save_all_states 为 True，则会保存所有状态，包括模型参数、优化器状态和学习率调度器状态，并将它们保存到一个名为 checkpoint_gpu_{rank}.pt 的文件中
        # 其中，rank 是当前进程的全局排名
        # 如果 rank 为 0，则会额外保存模型参数到一个名为 model.pt 的文件中，并使用 wandb_logger 记录模型参数到 W&B 平台上
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        # 如果设置了 cfg.save_all_states 为 False，则会仅保存模型参数到一个名为 model.pt 的文件中，并使用 wandb_logger 记录模型参数到 W&B 平台上
        # 在保存模型参数之后，它会检查是否设置了 wandb_logger 并且设置了 cfg.save_artifacts 为 True，
        # 如果是，则会创建一个名为 {run_name}_E{epoch} 或 {run_name}_Final 的 W&B artifact，并将模型参数添加到 artifact 中，最后将 artifact 记录到 W&B 平台上
        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)
        #  如果设置了 cfg.dali 为 True，则会在每个 epoch 结束后调用 train_loader.reset() 方法重置数据加载器
        if cfg.dali:
            train_loader.reset()


    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)
        
        if wandb_logger and cfg.save_artifacts:
            artifact_name = f"{run_name}_Final"
            model = wandb.Artifact(artifact_name, type='model')
            model.add_file(path_module)
            wandb_logger.log_artifact(model)





# 解析命令行参数，并调用 main 函数开始训练
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    # 分布式Arcface Pytorch训练

    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus.' 
    						 'Automatically set by torch.distributed.launch')
    parser.add_argument("--config", type=str, default="configs/wf2601_r50", help="py config file")

    # configs/glint360k_r50

    main(parser.parse_args())




