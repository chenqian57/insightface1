#!/usr/bin/env bash
# - `--model-prefix pretrained_models/r180-cosface-cmswbi/model`: 模型文件的路径前缀。
# - `--model-epoch 0`: 模型文件的epoch数。
# - `--gpu 0`: 指定使用的GPU ID。
# - `--target IJBC`: 指定测试的数据集，可以是`IJBC`或`IJBB`。
# - `--job r180cmswbi`: 指定作业名称。
# - `> r180cmswbi.log 2>&1 &`: 将脚本的输出结果重定向到文件`r180cmswbi.log`中，并在后台运行该命令。
python -u IJB_11.py --model-prefix pretrained_models/r180-cosface-cmswbi/model --model-epoch 0 --gpu 0 --target IJBC --job r180cmswbi > r180cmswbi.log 2>&1 &

