# import cv2
# import numpy as np
#
# # 读取图像
# image = cv2.imread('F:/demo_images/03_cus.png', cv2.IMREAD_GRAYSCALE)
#
# # 二值化图像
# _, binary_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
#
# # 设定连续黑色像素点的阈值
# threshold_value = 100  # 根据需求调整阈值
#
# # 寻找黑色像素点的坐标
# black_pixels = np.column_stack(np.where(binary_image == 0))
#
# # 处理连续的黑色像素点
# consecutive_black_count = 0
# for i in range(1, len(black_pixels)):
#     current_pixel = black_pixels[i]
#     previous_pixel = black_pixels[i - 1]
#
#     # 计算相邻两点之间的距离
#     distance = np.linalg.norm(np.array(current_pixel) - np.array(previous_pixel))
#
#     # 如果距离超过阈值，将之前的黑色像素点个数记录并置零
#     if distance > 1:
#         if consecutive_black_count > threshold_value:
#             # 将这些黑色像素点变为白色
#             start_idx = i - consecutive_black_count
#             end_idx = i
#             for idx in range(start_idx, end_idx):
#                 row, col = black_pixels[idx]
#                 binary_image[row, col] = 255
#         consecutive_black_count = 0
#     else:
#         consecutive_black_count += 1
#
# # 处理可能在图像边界的情况
# if consecutive_black_count > threshold_value:
#     start_idx = len(black_pixels) - consecutive_black_count
#     end_idx = len(black_pixels)
#     for idx in range(start_idx, end_idx):
#         row, col = black_pixels[idx]
#         binary_image[row, col] = 255
#
# # 保存处理好的图片到指定路径
# cv2.imwrite('F:/demo_images/output_images_3.png', binary_image)
#
# # 显示或保存结果
# cv2.imshow('Processed Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # 读取图像
# image = cv2.imread('F:/demo_images/03_cus.png', cv2.IMREAD_GRAYSCALE)
#
# # 二值化图像
# _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#
# # 设定连续黑色像素点的阈值
# horizontal_threshold = 100  # 根据需求调整阈值（水平方向）
# vertical_threshold = 100  # 根据需求调整阈值（垂直方向）
#
# # 寻找黑色像素点的坐标
# black_pixels = np.column_stack(np.where(binary_image == 0))
#
# # 处理水平方向上的连续黑色像素点
# for col in range(binary_image.shape[1]):
#     consecutive_black_count = 0
#     for row in range(binary_image.shape[0]):
#         if binary_image[row, col] == 0:
#             consecutive_black_count += 1
#             if consecutive_black_count > horizontal_threshold:
#                 binary_image[:, col] = 255
#                 break
#         else:
#             consecutive_black_count = 0
#
# # 处理垂直方向上的连续黑色像素点
# for row in range(binary_image.shape[0]):
#     consecutive_black_count = 0
#     for col in range(binary_image.shape[1]):
#         if binary_image[row, col] == 0:
#             consecutive_black_count += 1
#             if consecutive_black_count > vertical_threshold:
#                 binary_image[row, :] = 255
#                 break
#         else:
#             consecutive_black_count = 0
#
# # 保存处理好的图片到指定路径
# cv2.imwrite('F:/demo_images/output_images_4.png', binary_image)
#
# # 显示或保存结果
# cv2.imshow('Processed Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# def mark_consecutive_pixels(image, row, col, marked, count):
#     if row < 0 or row >= image.shape[0] or col < 0 or col >= image.shape[1]:
#         return count
#
#     if marked[row, col] or image[row, col] != 0:
#         return count
#
#     marked[row, col] = True
#     count += 1
#
#     # 检查上、下、左、右四个方向
#     count = mark_consecutive_pixels(image, row - 1, col, marked, count)
#     count = mark_consecutive_pixels(image, row + 1, col, marked, count)
#     count = mark_consecutive_pixels(image, row, col - 1, marked, count)
#     count = mark_consecutive_pixels(image, row, col + 1, marked, count)
#
#     return count
#
# # 读取图像
# image = cv2.imread('F:/demo_images/03_cus.png', cv2.IMREAD_GRAYSCALE)
#
# # 二值化图像
# _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#
# # 设定连续黑色像素点的阈值
# threshold_value = 100  # 根据需求调整阈值
#
# # 创建一个标记矩阵，记录像素是否已经被处理
# marked = np.zeros_like(binary_image, dtype=bool)
#
# # 遍历图像中的每个像素
# for row in range(binary_image.shape[0]):
#     for col in range(binary_image.shape[1]):
#         # 如果是黑色像素且未被处理过
#         if binary_image[row, col] == 0 and not marked[row, col]:
#             # 初始化计数器
#             count = 0
#             # 标记连续的黑色像素点，并获取计数值
#             count = mark_consecutive_pixels(binary_image, row, col, marked, count)
#             # 如果计数值超过阈值，将这些黑色像素点变为白色
#             if count > threshold_value:
#                 marked[row - count + 1:row + 1, col] = True
#                 binary_image[row - count + 1:row + 1, col] = 255
#
# # 保存处理好的图片到指定路径
# cv2.imwrite('F:/demo_images/output_images_5.png', binary_image)
#
# # 显示或保存结果
# cv2.imshow('Processed Image', binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def mark_consecutive_pixels(image, row, col, marked, threshold_value):
    stack = [(row, col)]
    count = 0

    while stack:
        r, c = stack.pop()
        if r < 0 or r >= image.shape[0] or c < 0 or c >= image.shape[1]:
            continue

        if marked[r, c] or image[r, c] > threshold_value:
            continue

        marked[r, c] = True
        count += 1

        # 将相邻未处理的黑色像素点加入栈中
        stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    return count

# 读取图像
image = cv2.imread('F:/demo_images/03_cus.png', cv2.IMREAD_GRAYSCALE)

# 二值化图像
_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

# 设定连续黑色像素点的阈值
threshold_value = 100  # 根据需求调整阈值

# 创建一个标记矩阵，记录像素是否已经被处理
marked = np.zeros_like(binary_image, dtype=bool)

# 遍历图像中的每个像素
for row in range(binary_image.shape[0]):
    for col in range(binary_image.shape[1]):
        # 如果是黑色像素且未被处理过
        if binary_image[row, col] == 0 and not marked[row, col]:
            # 标记连续的黑色像素点，并获取计数值
            count = mark_consecutive_pixels(binary_image, row, col, marked, threshold_value)
            # 如果计数值超过阈值，将这些黑色像素点变为白色
            if count > threshold_value:
                marked[row - count + 1:row + 1, col] = True
                binary_image[row - count + 1:row + 1, col] = 255

# 保存处理好的图片到指定路径
cv2.imwrite('F:/demo_images/output_images_5.png', binary_image)

# 显示或保存结果
cv2.imshow('Processed Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
