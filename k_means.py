# -*- coding:utf-8 -*-
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from skimage import color

# 定义函数: 加载图像, 并进行规范化
def load_data(filepath):
    f = open(filepath, 'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像的尺寸
    width, height = img.size
    for i in range(width):
        for j in range(height):
            # 得到点(i, j)的三个通道值
            c1, c2, c3 = img.getpixel((i, j))
            data.append([(c1+1)/256.0, (c2+1)/256.0, (c3+1)/256.0])
    f.close()
    return np.mat(data), width, height


# 加载图像，得到规范化的结果n_data和图像尺寸
n_data, width, height = load_data('./Thanos.jpg')

# 用K-Means对图像进行聚类
kmeans = KMeans(n_clusters=4)
# kmeans =KMeans(n_clusters=16)
label = kmeans.fit_predict(n_data)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])

# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label) * 255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
images = image.fromarray(label_color)
images.save('mark_color2.jpg')


# # 创建个新图像img，用来保存图像聚类压缩后的结果
# img = image.new('RGB', (width, height))
# for x in range(width):
#     for y in range(height):
#         # 提取 (x, y) 点对应的聚类中心的通道值
#         c1 = kmeans.cluster_centers_[label[x, y], 0]
#         c2 = kmeans.cluster_centers_[label[x, y], 1]
#         c3 = kmeans.cluster_centers_[label[x, y], 2]
#         img.putpixel((x, y), (int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
#
# img.save('new_4.jpg')
