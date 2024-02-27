# import cv2
# import numpy as np
#
# # 定义 process_connected_component 函数
# def process_connected_component(binary_image, processed, i, j, continuous_threshold):
#     stack = [(i, j)]
#
#     while stack:
#         i, j = stack.pop()
#         if 0 <= i < height and 0 <= j < width and binary_image[i, j] == 0 and processed[i, j] == 0:
#             binary_image[i, j] = 255  # 将黑色像素点变为白色
#             processed[i, j] = 1
#
#             # 将相邻的未处理的黑色像素点加入栈
#             stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
#
# # 读取图像
# image = cv2.imread('F:/demo_images/03_cus.png')
#
# # 将图像转换为灰度
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 使用阈值分割，将图像二值化
# _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#
# # 初始化 processed 标记矩阵
# processed = np.zeros_like(binary_image)
#
# # 定义连续值的阈值
# continuous_threshold = 10
#
# # 获取图像的高度和宽度
# height, width = binary_image.shape
#
# while True:
#     found_black_pixel = False
#
#     # 遍历图像的每个像素
#     for i in range(height):
#         for j in range(width):
#             # 如果当前像素是黑色，并且未被处理
#             if binary_image[i, j] == 0 and processed[i, j] == 0:
#                 found_black_pixel = True
#                 process_connected_component(binary_image, processed, i, j, continuous_threshold)
#
#     # 如果没有找到黑色像素点，跳出循环
#     if not found_black_pixel:
#         break
#
# # 保存处理好的图片到指定路径
# output_path = 'F:/demo_images/your_output_path_5.jpg'
# cv2.imwrite(output_path, binary_image)
#
# # 显示结果
# cv2.imshow('Result', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 定义 process_connected_component 函数
def process_connected_component(binary_image, processed, i, j, continuous_threshold):
    stack = [(i, j)]

    while stack:
        i, j = stack.pop()
        if 0 <= i < height and 0 <= j < width and binary_image[i, j] == 0 and processed[i, j] == 0:
            binary_image[i, j] = 255  # 将黑色像素点变为白色
            processed[i, j] = 1

            # 将相邻的未处理的黑色像素点加入栈
            stack.extend([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

# 读取图像
image = cv2.imread('F:/demo_images/03_cus.png')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 初始化 processed 标记矩阵
processed = np.zeros_like(gray)

# 定义连续值的阈值
continuous_threshold = 50

# 获取图像的高度和宽度
height, width = gray.shape

while True:
    found_black_pixel = False

    # 遍历图像的每个像素
    for i in range(height):
        for j in range(width):
            # 如果当前像素是黑色，并且未被处理
            if gray[i, j] == 0 and processed[i, j] == 0:
                found_black_pixel = True
                process_connected_component(gray, processed, i, j, continuous_threshold)

    # 如果没有找到黑色像素点，跳出循环
    if not found_black_pixel:
        break

# 保存处理好的图片到指定路径
output_path = 'F:/demo_images/your_output_path_5.jpg'
cv2.imwrite(output_path, gray)

# 显示结果
cv2.imshow('Result', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()