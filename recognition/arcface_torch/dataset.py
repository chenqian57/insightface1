import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn






# 定义了一个名为 get_dataloader 的函数，用于创建数据加载器
# 它根据数据集的类型创建不同的数据集实例，并根据是否使用 DALI 加速数据加载创建不同的数据加载器
# 在创建数据加载器时，它还设置了一些参数，包括本地进程的排名、批次大小、工作进程数、是否将数据加载到 GPU 内存中等

# 接受多个参数，包括数据集的根目录 root_dir、
# 本地进程的排名 local_rank、
# 批次大小 batch_size、
# 是否使用 DALI 加速数据加载 dali、
# 随机数种子 seed
# 工作进程数 num_workers
def get_dataloader(
    root_dir,      # 数据集的根目录，判断数据集类型
    local_rank,    # 
    batch_size,
    dali = False,
    seed = 2048,
    # 2, 44
    num_workers = 2,
    ) -> Iterable:


    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # 如果是 "synthetic"，则表示使用合成数据集，创建一个 SyntheticDataset 类的实例；
    # 如果存在 train.rec 和 train.idx 文件，则表示使用 Mxnet RecordIO 数据集，创建一个 MXFaceDataset 类的实例；
    # 否则，表示使用图像文件夹数据集，创建一个 ImageFolder 类的实例，
    # 并定义了一个 transforms.Compose 对象，用于对图像进行预处理
    # Synthetic
    if root_dir == "synthetic":
        train_set = SyntheticDataset()
        dali = False

    # Mxnet RecordIO
    elif os.path.exists(rec) and os.path.exists(idx):
        train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)

    # Image Folder
    # 创建一个 ImageFolder 类的实例（创建一个ImageFolder数据集对象），并定义一个 transforms.Compose 对象，用于对图像进行预处理
    else:
        transform = transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        train_set = ImageFolder(root_dir, transform)


    # 如果 dali 为 True，则表示使用 DALI 加速数据加载，调用 dali_data_iter 函数创建一个 DALI 数据加载器，并返回它
    # DALI
    if dali:
        return dali_data_iter(
            batch_size=batch_size, rec_file=rec, idx_file=idx,
            num_threads=2, local_rank=local_rank)

    # 否则，使用 PyTorch 的 DistributedSampler 类创建一个分布式采样器，并使用 PyTorch 的 DataLoaderX 类创建一个数据加载器，并返回它
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    # 在创建数据加载器时，它还设置了一些参数，包括本地进程的排名、批次大小、工作进程数、是否将数据加载到 GPU 内存中等
    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader








# 实现了一个生成器
# 接受多个参数，包括一个生成器 generator、本地进程的排名 local_rank 和 最大预取数量 max_prefetch
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        # 类的构造函数。创建了一个 Queue.Queue 对象，并将 max_prefetch 作为参数传递给它
        # 然后，它将 generator 和 local_rank 存储到类的成员变量中，并将当前线程设置为守护线程，并启动线程
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    # 在线程的 run 方法中，它首先调用 torch.cuda.set_device 函数将当前线程的 GPU 设备 ID 设置为 local_rank，
    # 以确保在多 GPU 环境下每个线程使用不同的 GPU 设备
    # 然后，它使用 for 循环遍历 generator，并将生成的数据放入队列中
    # 最后，它向队列中放入一个 None 对象，表示数据生成结束
    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    # 定义了 next、__next__ 和 __iter__ 方法，用于实现生成器的迭代
    # 在 next 方法中，它从队列中获取下一个数据项，并检查是否为 None，如果是，则抛出 StopIteration 异常，否则返回数据项
    # 在 __next__ 方法中，它调用 next 方法
    # 在 __iter__ 方法中，它返回类的实例本身，以支持迭代操作
    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self










# 定义了一个名为 DataLoaderX 的类，它继承自 PyTorch 的 DataLoader 类，并重写了 __iter__ 和 __next__ 方法，以支持多 GPU 环境下的数据加载。
# 使用了后台数据预取线程和 CUDA 流，以加速数据加载和移动

# 定义了一个名为 DataLoaderX 的类，它继承自 PyTorch 的 DataLoader 类
# 并重写了 __iter__ 和 __next__ 方法，以支持多 GPU 环境下的数据加载
class DataLoaderX(DataLoader):
    # 在类的构造函数中，它接受一个 local_rank 参数，表示本地进程的排名
    # 在 __init__ 方法中，它调用了父类的构造函数，并创建了一个 CUDA 流和一个本地排名变量，并将它们存储到类的成员变量中
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    # 在 __iter__ 方法中，它首先调用父类的 __iter__ 方法，获取一个数据迭代器
    # 然后，它将数据迭代器传递给 BackgroundGenerator 类的构造函数，创建一个后台数据预取线程
    # 并将预取线程的迭代器作为新的数据迭代器
    # 最后，它调用 preload 方法预取一个数据批次，并返回自身
    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self


    # 在 preload 方法中，它从预取线程的迭代器中获取一个数据批次，并将数据批次中的每个张量都移动到本地进程对应的 GPU 设备上
    # 在移动张量时，它使用了 CUDA 流，以避免数据移动和计算之间的竞争
    # 最后，它将移动后的数据批次存储到类的成员变量中
    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)


    # 在 __next__ 方法中，它等待 CUDA 流执行完毕，并返回存储在类的成员变量中的数据批次
    # 如果数据迭代器已经到达末尾，则抛出 StopIteration 异常
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch











# 名为 MXFaceDataset 的类，它继承自 PyTorch 的 Dataset 类，并实现了一个数据集
class MXFaceDataset(Dataset):
    # 它接受两个参数，包括数据集的根目录 root_dir 和 本地进程的排名 local_rank
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # 在类的构造函数中，它首先定义了一个 transforms.Compose 对象，用于对图像进行预处理
        # 然后，它将 root_dir 和 local_rank 存储到类的成员变量中，并根据 root_dir 中的文件名创建了一个 Mxnet RecordIO 数据集实例，并读取了数据集的头文件
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        # 如果头文件中的标志大于 0，则表示数据集中包含标签信息，它将标签信息存储到类的成员变量中，并创建一个索引数组；
        # 否则，它将数据集中的所有键存储到索引数组中
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
    # 在 __getitem__ 方法中，它首先根据索引获取数据集中的一条数据，并解包数据集头文件和图像数据
    # 然后，它将标签转换为 PyTorch 张量，并将图像数据解码为 NumPy 数组
    # 接下来，它使用 transforms.Compose 对象对图像进行预处理，并将预处理后的图像数据和标签返回
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label
    # 在 __len__ 方法中，它返回数据集的长度，即索引数组的长度
    def __len__(self):
        return len(self.imgidx)










# 名为 SyntheticDataset 的类，它继承自 PyTorch 的 Dataset 类，并实现了一个合成数据集。它没有接受任何参数
class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000










# 接受多个参数，包括批量大小 batch_size、记录文件路径 rec_file、索引文件路径 idx_file、线程数 num_threads、
# 初始填充大小 initial_fill、是否随机打乱 random_shuffle、预取队列深度 prefetch_queue_depth、
# 本地进程排名 local_rank、名称 name、均值 mean 和标准差 std
def dali_data_iter(
    batch_size: int, rec_file: str, idx_file: str, num_threads: int,
    initial_fill=32768, random_shuffle=True,
    prefetch_queue_depth=1, local_rank=0, name="reader",
    mean=(127.5, 127.5, 127.5), 
    std=(127.5, 127.5, 127.5)):
    """
    Parameters:
    ----------
    initial_fill: int
        Size of the buffer that is used for shuffling. If random_shuffle is False, this parameter is ignored.

    """
    rank: int = distributed.get_rank()
    world_size: int = distributed.get_world_size()
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator

    pipe = Pipeline(
        batch_size=batch_size, num_threads=num_threads,
        device_id=local_rank, prefetch_queue_depth=prefetch_queue_depth, )
    condition_flip = fn.random.coin_flip(probability=0.5)
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file, index_path=idx_file, initial_fill=initial_fill, 
            num_shards=world_size, shard_id=rank,
            random_shuffle=random_shuffle, pad_last_batch=False, name=name)
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.crop_mirror_normalize(
            images, dtype=types.FLOAT, mean=mean, std=std, mirror=condition_flip)
        pipe.set_outputs(images, labels)
    pipe.build()
    return DALIWarper(DALIClassificationIterator(pipelines=[pipe], reader_name=name, ))













@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict['data'].cuda()
        tensor_label: torch.Tensor = data_dict['label'].cuda().long()
        tensor_label.squeeze_()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def reset(self):
        self.iter.reset()
