# 自行添加，有错误

import cv2
import numpy as np
import os
import insightface

from insightface.app import FaceAnalysis

# 创建FaceAnalysis对象
fa = FaceAnalysis()

# 加载模型
fa.prepare(ctx_id=-1, nms=0.4)


# 加载模型
model = insightface.model_zoo.get_model('arcface_r100_v1')

# 加载测试数据集
test_dir = '/mnt/ssd/qiujing/arcface/lfw/lfw112/112×112'  # /path/to/test/dataset 
test_imgs = []
test_labels = []
for label_name in os.listdir(test_dir):
    label_dir = os.path.join(test_dir, label_name)
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_imgs.append(img)
        test_labels.append(label_name)

# 对测试数据集进行预测
pred_labels = []
for img in test_imgs:
    embedding = fa.get_embedding(img)   # embedding = model.get_embedding(img)
    pred_label = np.argmax(model.compute_sim(embedding, model.threshold))
    pred_labels.append(str(pred_label))

# 计算准确率和误识率
correct_count = 0
false_count = 0
for i in range(len(test_labels)):
    if test_labels[i] == pred_labels[i]:
        correct_count += 1
    else:
        false_count += 1

accuracy = correct_count / len(test_labels)
false_rate = false_count / len(test_labels)

print('Accuracy:', accuracy)
print('False Rate:', false_rate)
