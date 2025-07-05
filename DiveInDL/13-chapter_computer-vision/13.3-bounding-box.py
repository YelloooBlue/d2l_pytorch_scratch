import torch
from matplotlib import pyplot as plt

# 图片坐标系转换
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    # 输入边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # Rectangle需要((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )


if __name__ == "__main__":

    # 读取图片
    img = plt.imread('DiveInDL/img/catdog.jpg')
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    # plt.show()

    # bbox是边界框的英文缩写
    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

    # 测试坐标转换
    boxes = torch.tensor([dog_bbox, cat_bbox])
    print("相等：", torch.equal(box_center_to_corner(box_corner_to_center(boxes)), boxes))

    # 绘制边界框
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    plt.show()
