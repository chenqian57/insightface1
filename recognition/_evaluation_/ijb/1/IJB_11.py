# coding: utf-8

from mxnet import np,npx
#npx.set_np()
import os
import numpy as np
#import cPickle
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import cv2
import sys
import glob

# /home/qiujing/cqwork/insightface/recognition/_evaluation_/ijb/1
# sys.path.append('./recognition')

sys.path.append('/home/qiujing/cqwork/insightface/recognition/_evaluation_/ijb/1')
# from embedding import Embedding
from embedding_pytorch import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")  

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='/mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth', help='path to load model.')
# parser.add_argument('--model-prefix', default='/home/qiujing/cqwork/insightface/recognition/_evaluation_/megaface/MS1MV2-mxnet-r100-ii/model', help='path to load model.')
# /mnt/ssd/qiujing/arcface/megaface/MS1MV2-mxnet-r100-ii
# /mnt/ssd/qiujing/arcface/eval/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth
# 模型文件的路径前缀
# 

parser.add_argument(
        "-b1",
        "--backbone",
        default="ir50",
        help="主干特征提取网络的选择, mobilefacenet,mobilenetv1,iresnet18,iresnet34,iresnet50,iresnet100,iresnet200,resnet50",
        # # ir18, ir34, ir50, ir100, ir200
        # mobilefacenet, mobilenetv1, iresnet18, iresnet34, iresnet50, iresnet100, iresnet200, resnet34, resnet50, resnet101, resnet152, vit, iresnet2060
    )




parser.add_argument('--model-epoch', default=1, type=int, help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--batch-size', default=32, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
# 要测试的数据集

args = parser.parse_args()



target = args.target
backbone = args.backbone
model_path = args.model_prefix  # 模型文件的路径前缀
gpu_id = args.gpu
epoch = args.model_epoch
use_norm_score = True # if Ture, TestMode(N1)      # 是否使用归一化得分进行测试
# True，False
use_detector_score = False # if Ture, TestMode(D1)  # 是否使用检测器得分进行测试
use_flip_test = True # if Ture, TestMode(F1)       # 是否进行水平翻转测试
job = args.job


# 从指定路径读取文件，并返回模板和媒体列表
def read_template_media_list(path):
    #ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values

    # 转换为NumPy数组。
    # 从数组中提取第2列和第3列，并将它们转换为整数类型
    templates = ijb_meta[:,1].astype(np.int)
    medias = ijb_meta[:,2].astype(np.int)
    return templates, medias

# In[ ]:


# 从指定路径读取文件，并返回模板对列表、标签列表
def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    #print(pairs.shape)
    #print(pairs[:, 0].astype(np.int))
    # 从数组中提取第1列、第2列和第3列，并将它们转换为整数类型
    t1 = pairs[:,0].astype(np.int)
    t2 = pairs[:,1].astype(np.int)
    label = pairs[:,2].astype(np.int)
    return t1, t2, label

# In[ ]:


# 从指定路径读取文件，并返回图像特征
def read_image_feature(path):
    with open(path, 'rb') as fid:
        # 使用了Python的pickle模块，因此读取的文件必须是使用pickle模块序列化的对象
        img_feats = pickle.load(fid)
    return img_feats

# In[ ]:








# 从指定路径读取图像列表文件，并使用人脸识别模型提取每个图像的特征
def get_image_feature(img_path, img_list_path, backbone, model_path, epoch, gpu_id):

    # 打开指定路径的图像列表文件
    img_list = open(img_list_path)



    embedding = Embedding(backbone, model_path, epoch, gpu_id)
    # 逐行读取图像列表文件，并使用cv2.imread方法读取每个图像
    
    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_feats = []

    # 从一个文件中读取每行的文本，提取人脸关键点坐标和人脸评分，并将每个图像的特征和人脸评分存储在数组中
    for img_index, each_line in enumerate(files):
        if img_index%500==0:
          print('processing', img_index)
        
        # 将当前文本按照空格进行分割，得到列表 name_lmk_score，包括图像路径、人脸关键点坐标和人脸评分
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])

        # 使用cv2.imread函数读取图像
        img = cv2.imread(img_name)

        # # 将列表name_lmk_score中的第2个到倒数第2个元素（即人脸关键点坐标）转换为一个numpy数组lmk，并将其存储在变量lmk中
        # lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)

        # # 将人脸关键点坐标转换为一个5*2的数组lmk
        # lmk = lmk.reshape( (5,2) )
        # # 该函数从每行中提取出人脸关键点坐标，将图像和人脸关键点坐标作为参数传递给embedding.get方法，以提取该图像的特征
        # # 将特征添加到img_feats数组中
        img_feats.append(embedding.get(img))
        
        # 将人脸评分添加到 faceness_scores 数组中
        # 将最后一个元素添加到faceness_scores数组中
        faceness_scores.append(name_lmk_score[-1])
    # 该函数将 所有图像的特征 和 人脸评分 分别存储在img_feats和faceness_scores数组中，并返回这两个数组。
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    #img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    #faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    # 返回两个数组，分别存储了所有图像的特征和人脸评分
    return img_feats, faceness_scores
















# # 从指定路径读取图像列表文件，并使用人脸识别模型提取每个图像的特征
# def get_image_feature(img_path, img_list_path, backbone, model_path, epoch, gpu_id):
#     # 打开指定路径的图像列表文件
#     img_list = open(img_list_path)
#     embedding = Embedding(backbone, model_path, epoch, gpu_id)
#     # 逐行读取图像列表文件，并使用cv2.imread方法读取每个图像
    
#     files = img_list.readlines()
#     print('files:', len(files))
#     faceness_scores = []
#     img_feats = []

#     # 从一个文件中读取每行的文本，提取人脸关键点坐标和人脸评分，并将每个图像的特征和人脸评分存储在数组中
#     for img_index, each_line in enumerate(files):
#         if img_index%500==0:
#           print('processing', img_index)
        
#         # 将当前文本按照空格进行分割，得到列表 name_lmk_score，包括图像路径、人脸关键点坐标和人脸评分
#         name_lmk_score = each_line.strip().split(' ')
#         img_name = os.path.join(img_path, name_lmk_score[0])

#         # 使用cv2.imread函数读取图像
#         img = cv2.imread(img_name)

#         # 将列表name_lmk_score中的第2个到倒数第2个元素（即人脸关键点坐标）转换为一个numpy数组lmk，并将其存储在变量lmk中
#         lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)

#         # 将人脸关键点坐标转换为一个5*2的数组lmk
#         lmk = lmk.reshape( (5,2) )
#         # 该函数从每行中提取出人脸关键点坐标，将图像和人脸关键点坐标作为参数传递给embedding.get方法，以提取该图像的特征
#         # 将特征添加到img_feats数组中
#         img_feats.append(embedding.get(img,lmk))
        
#         # 将人脸评分添加到 faceness_scores 数组中
#         # 将最后一个元素添加到faceness_scores数组中
#         faceness_scores.append(name_lmk_score[-1])
#     # 该函数将 所有图像的特征 和 人脸评分 分别存储在img_feats和faceness_scores数组中，并返回这两个数组。
#     img_feats = np.array(img_feats).astype(np.float32)
#     faceness_scores = np.array(faceness_scores).astype(np.float32)

#     #img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
#     #faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
#     # 返回两个数组，分别存储了所有图像的特征和人脸评分
#     return img_feats, faceness_scores












# In[ ]:




# 目的：从图像特征中计算模板特征
# 计算分三步完成。首先，对图像特征进行L2归一化处理。
# 其次，通过聚合来自同一媒体的图像特征来计算媒体特征。
# 第三，通过聚合来自同一模板的媒体特征来计算模板特征
# ' img_feat '是一个形状为' (number_image, feats_dim) '的NumPy数组，其中' number_image '是图像的数量，' feats_dim '是图像特征的维度。
# ' templates '是一个形状为' (number_image，) '的NumPy数组，其中包含每个图像的模板template id。
# ' medias '是一个形状为' (number_image，) '的NumPy数组，其中包含每个图像的媒体medias id。
def image2template_feature(img_feats = None, templates = None, medias = None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================   
    # 为了计算模板特征，该函数首先从' templates '数组中提取唯一的模板id。
    # 对于每个唯一的模板ID，它查找属于该模板的图像的索引。然后提取这些图像的媒体特征并对其进行求和，得到模板特征。
    # 然后对模板特征进行L2规范化。 
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        # 为了计算媒体特征，该函数首先从' media '数组中提取唯一的媒体id
        # 对于每个唯一的媒体ID，它查找属于该媒体的图像的索引
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u,ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            # 如果只有一个镜像，则使用镜像特性作为媒体特性
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            # 如果有多个图像，则对图像特征进行平均，得到媒体特征。然后将媒体特征连接起来形成媒体特征向量。
            else: # image features from the same video will be aggregated into one feature，来自同一视频的图像特征将被聚合为一个特征
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))

        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0: 
            print('Finish Calculating {} template features.'.format(count_template))
    #template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    #print(template_norm_feats.shape)

    # 返回两个值:' template_norm_feat '和' unique_templates '。
    # ' template_norm_feat '是一个形状为' (number_template, feats_dim) '的NumPy数组，包含L2规范化模板特征。
    # ' unique_templates '是一个形状为' (number_template，) '的NumPy数组，其中包含唯一的模板id。
    return template_norm_feats, unique_templates

# In[ ]:





# 目的：计算模板对之间的相似度分数
# 1、使用模板ID从template_norm_feats数组中提取相应的模板特征。
# 2、计算模板特征对之间的余弦相似度分数。
def verification(template_norm_feats = None, unique_templates = None, p1 = None, p2 = None):
    # template_norm_feats是一个形状为(number_template, feats_dim)的NumPy数组，其中包含L2归一化的模板特征。
    # unique_templates是一个形状为(number_template,)的NumPy数组，其中包含唯一的模板ID。
    # p1和p2是形状为(number_pairs,)的NumPy数组，其中包含每个模板对的模板ID。
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    # 为了计算相似度分数，该函数首先创建一个字典，将每个模板ID映射到其在template_norm_feats数组中的索引。
    # 然后，它以batchsize大小的批次迭代模板ID对。
    # 对于每个批次，它从template_norm_feats数组中提取相应的模板特征，并计算模板特征对之间的余弦相似度分数。
    # 相似度分数然后存储在一个NumPy数组中。
    template2id = np.zeros((max(unique_templates)+1,1),dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    
    score = np.zeros((len(p1),))   # save cosine distance between pairs 

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000 # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        # 在计算过程中，该函数每10个批次打印一条消息，以指示进度
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    # 该函数返回一个形状为(number_pairs,)的NumPy数组，其中包含模板对之间的相似度分数。
    return score






# In[ ]:

# 目的：计算模板对之间的相似度分数
# 计算分为两个步骤。首先，使用模板ID从template_norm_feats数组中提取相应的模板特征。
# 其次，计算模板特征对之间的余弦相似度分数。
def verification2(template_norm_feats = None, unique_templates = None, p1 = None, p2 = None):
# template_norm_feats是一个形状为(number_template, feats_dim)的NumPy数组，其中包含L2归一化的模板特征。
# unique_templates是一个形状为(number_template,)的NumPy数组，其中包含唯一的模板ID。
# p1和p2是形状为(number_pairs,)的NumPy数组，其中包含每个模板对的模板ID。
  template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
  # 为了计算相似度分数，该函数首先创建一个字典，将每个唯一的模板ID映射到其在template_norm_feats数组中的索引。
  for count_template, uqt in enumerate(unique_templates):
      template2id[uqt] = count_template
  score = np.zeros((len(p1),))  # save cosine distance between pairs

  total_pairs = np.array(range(len(p1)))
  batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
  # 然后，它以batchsize大小的批次迭代模板ID对
  sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
  total_sublists = len(sublists)
  # 对于每个批次，它从template_norm_feats数组中提取相应的模板特征，并计算模板特征对之间的余弦相似度分数。相似度分数然后存储在一个NumPy数组中。
  for c, s in enumerate(sublists):
      feat1 = template_norm_feats[template2id[p1[s]]]
      feat2 = template_norm_feats[template2id[p2[s]]]
      similarity_score = np.sum(feat1 * feat2, -1)
      score[s] = similarity_score.flatten()
      # 在计算过程中，该函数每10个批次打印一条消息，以指示进度。
      # 为了避免内存限制，该函数将模板对分成小批次进行处理，而不是一次性处理所有模板对
      if c % 10 == 0:
          print('Finish {}/{} pairs.'.format(c, total_sublists))
  # 该函数返回一个形状为(number_pairs,)的NumPy数组，其中包含模板对之间的相似度分数。
  return score




def read_img(image_path):

  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  return img  # numpy数组




def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

# # Step1: Load Meta Data

# In[ ]:



# 使用assert语句检查目标是否为'IJBC'或'IJBB'，以确保目标有效
assert target=='IJBC' or target=='IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id 
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
# 从文件中读取模板和媒体列表。该文件位于目标的'meta'目录中，文件名为'%s_face_tid_mid.txt'，其中'%s'是目标的小写版本。
# 模板和媒体列表分别存储在'templates'和'medias'变量中
# templates, medias = read_template_media_list(os.path.join('%s/meta'%target, '%s_face_tid_mid.txt'%target.lower()))
# %target.lower() '%s'是目标的小写版本。
templates, medias = read_template_media_list(os.path.join('/mnt/ssd/qiujing/arcface/ijb','%s/meta'%target, '%s_face_tid_mid.txt'%target.lower()))

# 模板和媒体列表的格式如下：每行包含一个图像名称、一个模板ID和一个媒体ID，由空格分隔。
# 模板ID用于将属于同一人的图像分组，而媒体ID用于标识组内的特定图像
stop = timeit.default_timer()

# 最后，该脚本打印出读取模板和媒体列表所需的时间。
# 时间使用'timeit'模块计算，该模块提供了一种简单的方法来计时小的Python代码片段。
print('Time: %.2f s. ' % (stop - start))

# In[ ]:


# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
# 读取模板对，以进行模板对比验证
start = timeit.default_timer()
# p1, p2, label = read_template_pair_list(os.path.join('%s/meta'%target, '%s_template_pair_label.txt'%target.lower()))
# 首先定义了模板对的格式，包括两个模板ID和一个标签，指示模板是否属于同一人。
# 模板ID表示为'tid_1'和'tid_2'，而标签表示为'label'。

# 从文件中读取模板对。该文件位于目标的'meta'目录中，文件名为'%s_template_pair_label.txt'，其中'%s'是目标的小写版本。
# 模板对存储在'p1'、'p2'和'label'变量中
# 模板对文件的格式如下：每行包含两个模板ID和一个标签，由空格分隔。如果模板属于同一人，则标签为'1'，否则为'0'
p1, p2, label = read_template_pair_list(os.path.join('/mnt/ssd/qiujing/arcface/ijb', '%s/meta'%target, '%s_template_pair_label.txt'%target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:


# =============================================================
# load image features 
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
# img_path = './%s/loose_crop' % target
img_path = '/mnt/ssd/qiujing/arcface/ijb/%s/loose_crop' % target

# imgs = []






img_list_path = '/mnt/ssd/qiujing/arcface/ijb/%s/meta/%s_name_5pts_score.txt' % (target, target.lower())

# for line in open(img_list_path, 'r'):
#   img = read_img(img_path)
#   if img is None:
#     print('read error:', img_path)
#     continue
#   imgs.append(img)


# img = read_img(img_path)
# if img is None:
#   print('read error:', img_path)
# imgs.append(img)





img_feats, faceness_scores = get_image_feature(img_path, img_list_path, backbone, model_path, epoch, gpu_id)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))

# # Step3: Get Template Features

# In[ ]:


# =============================================================
# compute template features from image features.
# 从图像特征计算模板特征
# =============================================================
start = timeit.default_timer()
# ========================================================== 
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ========================================================== 
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）
# 对图像特征进行处理，并将其转换为模板特征，以便进行后续的模板对比验证


# 首先定义一些参数，例如是否使用翻转测试、是否使用特征规范化 和 是否使用人脸检测器得分等
# 根据参数对图像特征进行处理。
# 如果使用翻转测试，则将图像特征拼接在一起，否则只使用前一半的图像特征。
if use_flip_test:
    # concat --- F1
    #img_input_feats = img_feats 
    # add --- F2
    img_input_feats = img_feats[:,0:img_feats.shape[1]//2] + img_feats[:,img_feats.shape[1]//2:]
else:
    img_input_feats = img_feats[:,0:img_feats.shape[1]//2]

# 如果使用特征规范化，则对 图像特征 进行规范化，否则不进行规范化
if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))    

# 如果使用人脸检测器得分，则将图像特征 乘以人脸检测器得分，否则不进行处理
if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    #img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
    img_input_feats = img_input_feats * faceness_scores[:,np.newaxis]
else:
    img_input_feats = img_input_feats

# 调用'image2template_feature'函数将图像特征转换为模板特征
# 该函数将 图像特征 与 模板列表 和 媒体列表 进行匹配，并将匹配的结果存储在'template_norm_feats'和'unique_templates'变量中。
template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
stop = timeit.default_timer()

# 打印出计算模板特征所需的时间
# 时间使用'timeit'模块计算
print('Time: %.2f s. ' % (stop - start))






# # Step 4: Get Template Similarity Scores
# 获取模板相似度评分

# In[ ]:


# =============================================================
# compute verification scores between template pairs.
# 计算模板对之间的验证分数
# =============================================================
start = timeit.default_timer()
# 调用'verification'函数执行模板对比验证。该函数将模板特征嵌入、唯一模板和模板对作为输入，并返回验证分数作为输出。
# 验证分数表示每个模板对中两个模板之间的相似度
score = verification(template_norm_feats, unique_templates, p1, p2)
# 验证完成后，该脚本使用'timeit'模块定义停止时间，并计算执行验证所需的时间
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# save_path = './%s_result' % target
save_path = '/mnt/ssd/qiujing/arcface/ijb/result/%s_result' % target



if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy"%job)
np.save(score_save_file, score)





# # Step 5: Get ROC Curves and TPR@FPR Table
# 得到 ROC曲线 和 TPR@FPR表

# In[ ]:

# 绘制IJB数据集上的ROC曲线，并输出每个方法在不同 假阳性率 下的 真阳性率
# 定义了一些参数，例如文件名、颜色映射 和 x轴标签
files = [score_save_file]
methods = []
scores = []
for file in files:
    # 使用'np.load'函数从文件中加载分数，并将分数存储在'scores'字典中
    methods.append(Path(file).stem)
    scores.append(np.load(file)) 

methods = np.array(methods)
scores = dict(zip(methods,scores))
colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
#x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]

# 使用'PrettyTable'模块创建一个表格，用于存储每个方法在不同假阳性率下的真阳性率
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    # 使用'roc_curve'函数计算每个方法的ROC曲线，并使用'auc'函数计算每个方法的AUC值。
    # 将ROC曲线绘制在图表上，并将 每个方法的名称 和 AUC值 添加到图例中。
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr
    plt.plot(fpr, tpr, color=colours[method], lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s"%(method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        #tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_row.append('%.2f' % (tpr[min_index]*100))
    
    # 使用'add_row'函数将每个方法在不同假阳性率下的真阳性率添加到表格中，并使用'savefig'函数将图表保存为PDF文件
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels) 
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True)) 
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
#plt.show()
fig.savefig(os.path.join(save_path, '%s.pdf'%job))
print(tpr_fpr_table)


# 总的来说，该脚本是在IJB数据集上绘制ROC曲线的一个重要步骤。它计算每个方法的ROC曲线和AUC值，并输出每个方法在不同假阳性率下的真阳性率
