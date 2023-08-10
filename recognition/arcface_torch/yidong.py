
import os
import shutil

source_dir = '/home/qiujing/face_data/webface26m/3/3_4/WebFace260M'
# /home/qiujing/face_data/webface26m/0/zong_0


# /home/qiujing/face_data/webface26m/0/0_0/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_1/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_2/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_3/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_4/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_5/WebFace260M
# /home/qiujing/face_data/webface26m/0/0_6/WebFace260M



# /home/qiujing/face_data/webface26m/1/1_0/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_1/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_2/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_3/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_4/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_5/WebFace260M
# /home/qiujing/face_data/webface26m/1/1_6/WebFace260M



# /home/qiujing/face_data/webface26m/2/2_0/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_1/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_2/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_3/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_4/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_5/WebFace260M
# /home/qiujing/face_data/webface26m/2/2_6/WebFace260M


# /home/qiujing/face_data/webface26m/3/3_0/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_1/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_2/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_3/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_4/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_5/WebFace260M
# /home/qiujing/face_data/webface26m/3/3_6/WebFace260M


# /home/qiujing/face_data/webface26m/4/4_0/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_1/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_2/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_3/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_4/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_5/WebFace260M
# /home/qiujing/face_data/webface26m/4/4_6/WebFace260M



# /home/qiujing/face_data/webface26m/5/5_0/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_1/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_2/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_3/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_4/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_5/WebFace260M
# /home/qiujing/face_data/webface26m/5/5_6/WebFace260M



# /home/qiujing/face_data/webface26m/6/6_0/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_1/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_2/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_3/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_4/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_5/WebFace260M
# /home/qiujing/face_data/webface26m/6/6_6/WebFace260M



# /home/qiujing/face_data/webface26m/7/7_0/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_1/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_2/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_3/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_4/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_5/WebFace260M
# /home/qiujing/face_data/webface26m/7/7_6/WebFace260M



# /home/qiujing/face_data/webface26m/8/8_0/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_1/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_2/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_3/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_4/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_5/WebFace260M
# /home/qiujing/face_data/webface26m/8/8_6/WebFace260M



# /home/qiujing/face_data/webface26m/9/9_0/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_1/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_2/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_3/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_4/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_5/WebFace260M
# /home/qiujing/face_data/webface26m/9/9_6/WebFace260M


dest_dir = '/mnt/ssd/qiujing/webface26m_zong2345'
# /mnt/ssd/qiujing/webface26m_zong01

# /mnt/ssd/qiujing/webface26m_zong2345
# /mnt/ssd/qiujing/webface26m_zong67
# /mnt/ssd/qiujing/webface26m_zong89






# 遍历 `source_dir` 目录下的所有文件和文件夹
# 如果它找到一个文件夹，它将使用 `shutil.move()` 函数将该文件夹移动到 `dest_dir` 目录中

for foldername in os.listdir(source_dir):
    if os.path.isdir(os.path.join(source_dir, foldername)):
        shutil.move(os.path.join(source_dir, foldername), os.path.join(dest_dir, foldername))











# # 打开文本文件
# with open('/home/qiujing/face_data/webface26m/0/01cls_train/wf26m01_cls_train.txt', 'r') as file:
#     # 使用readlines()方法读取所有行，并将其存储在一个列表中
#     lines = file.readlines()
# # /home/qiujing/face_data/webface26m/0/01cls_train/wf26m01_cls_train.txt
# # /home/qiujing/cqwork/insightface/recognition/arcface_torch/4.txt


# # 打印行数
# print("行数:", len(lines))





