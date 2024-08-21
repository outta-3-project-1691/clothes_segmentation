import torch
import numpy as np
import matplotlib.pyplot as plt

def show_segment(image_tensor):
    colors = {
        0: [128, 128, 128], # Gray
        1: [255, 0, 0],     # Red
        2: [0, 255, 0],     # Green
        3: [0, 0, 255],     # Blue
        4: [255, 255, 0],   # Yellow
        5: [255, 0, 255],   # Magenta
        6: [0, 255, 255],   # Cyan
    }

    # 3채널 컬러 이미지로 변환
    color_image = np.zeros((128, 128, 3), dtype=np.uint8)
    for value, color in colors.items():
        color_image[image_tensor == value] = color

    # 이미지를 보여주기
    plt.imshow(color_image)
    plt.axis('off')  # 축 제거
    plt.show()
