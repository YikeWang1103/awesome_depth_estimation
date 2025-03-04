import numpy as np
import copy
import cv2


def depth2colormap(depth_image, max_value, min_value, cmap):
    # 归一化深度图像
    normalized_depth_image = copy.deepcopy(depth_image)
    normalized_depth_image = (normalized_depth_image - min_value) / (max_value - min_value)
    normalized_depth_image = np.clip(normalized_depth_image, 0, 1)

    color_map = cmap(normalized_depth_image)[:, :, :3]  # 丢弃alpha通道
    color_map = (color_map * 255).astype(np.uint8)
    return color_map


def add_mask_overlay(rgb_image, mask, color=(0, 255, 0), alpha=0.4):
    """
    在RGB图像上添加半透明的mask效果
    
    参数:
        rgb_image: RGB格式的图像数组
        mask: 二值mask图像 (0或1)
        color: mask的颜色，默认为绿色 (B,G,R)
        alpha: 透明度，0-1之间，1为完全不透明
    """
    # 确保mask是二值图像
    mask = mask.astype(bool)
    
    # 创建彩色mask
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask] = color
    
    # 将mask叠加到原图上
    overlay = cv2.addWeighted(rgb_image, alpha, colored_mask, 1, 0)
    
    return overlay


def add_mask_contour(rgb_image, mask, color=(0, 255, 0), thickness=2):
    """
    在RGB图像上添加mask的轮廓效果
    
    参数:
        rgb_image: RGB格式的图像数组
        mask: 二值mask图像 (0或1)
        color: 轮廓颜色，默认为绿色 (B,G,R)
        thickness: 轮廓线条粗细
    """
    # 将mask转换为uint8类型
    mask = mask.astype(np.uint8)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # 复制原图，以免修改原始数据
    result = rgb_image.copy()
    
    # 在图像上绘制轮廓
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result


def mask2colormap(mask, cmap):
    mask_colormap = cmap(mask)[:, :, :3]  # 丢弃alpha通道
    mask_colormap = (mask_colormap * 255).astype(np.uint8)
    return mask_colormap