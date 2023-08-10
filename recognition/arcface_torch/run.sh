CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_v2.py $@

# $@ 表示传递给脚本的所有参数，这里将它们传递给了 train_v2.py 脚本