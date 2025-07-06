import torch
from matplotlib import pyplot as plt
torch.set_printoptions(2)  # 精简输出精度

# ======================================== 坐标转换 ========================================

# 来自13.3
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

# 来自13.3
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# 来自13.3
def bbox_to_rect(bbox, color):
    # 输入边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # Rectangle需要((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

# ======================================== 锚框生成 ========================================

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:]   # 从（batch_size, channels, height, width）中获取高度和宽度
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)

    # 以同一像素为中心的锚框的数量
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    """
        如果使用所有组合，复杂性很容易过高。
        这里我们只使用一个第一个「size」和所有「ratios」的组合，
        以及所有「sizes」和第一个「ratio」的组合。
    """

    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    """
        假设in_height=4, in_width=4, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]
        arange(in_height) = [0, 1, 2, 3]
        + offset_h = [0.5, 1.5, 2.5, 3.5]
        
        则steps_h=0.25, steps_w=0.25, boxes_per_pixel=5
        center_h = [0.125, 0.375, 0.625, 0.875]
        center_w = [0.125, 0.375, 0.625, 0.875]
    """
    
    # 生成中心点的网格  
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # print(shift_y.shape, shift_x.shape)  # (4, 4), (4, 4)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    """
        center_h = [0.125, 0.375, 0.625, 0.875]
        center_w = [0.125, 0.375, 0.625, 0.875]

        meshgrid生成的原始尺寸为：
        shift_y = [[0.125, 0.125, 0.125, 0.125],
                   [0.375, 0.375, 0.375, 0.375],
                   [0.625, 0.625, 0.625, 0.625],
                   [0.875, 0.875, 0.875, 0.875]]
        shift_x = [[0.125, 0.375, 0.625, 0.875],
                   [0.125, 0.375, 0.625, 0.875],
                   [0.125, 0.375, 0.625, 0.875],
                   [0.125, 0.375, 0.625, 0.875],
                   [0.125, 0.375, 0.625, 0.875]]

        reshape后变为一维向量：
        shift_y = [0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875]
        shift_x = [0.125, 0.125, 0.125, 0.125, 0.375, 0.375, 0.375, 0.375, 0.625, 0.625, 0.625, 0.625, 0.875, 0.875, 0.875, 0.875]
    """

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width  # 处理矩形输入
    
    """
        sizes = [0.75, 0.5, 0.25], size_tensor = [0.75, 0.5, 0.25]
        ratios = [1, 2, 0.5], ratio_tensor = [1, 2, 0.5]

        size_tensor * torch.sqrt(ratio_tensor[0])
             = [0.75, 0.5, 0.25] * sqrt(1) = [0.75, 0.5, 0.25]

        sizes[0] * torch.sqrt(ratio_tensor[1:])
             = 0.75 * [sqrt(2), sqrt(0.5)] = [1.06, 0.53]

        为什么要乘以sqrt？
            我们需要保证「正方形」和「矩形」锚框的面积相同，
            r = w / h
            w = h * r
            h = w / r
            w * h = size * size
            w^2 / r = size^2
            w = size * sqrt(r)

        cat = [0.75, 0.5, 0.25, 1.06, 0.53]
        第一部分为「正方形」锚框，大小分别为0.75, 0.5, 0.25，
        第二部分为「矩形」锚框，大小分别为1.06, 0.53。

        此时cat后的形状是相对于1*1坐标系
        如果直接应用在图上，会被拉伸，例如「正方形」的锚框会被拉伸成图像尺寸
        所以要乘以in_height / in_width来适应图像的「高宽比」。

    """
    
    # 和上面同理，但我们以高度为1基准，所以不用乘以in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    # print("w:", w)
    # print("h:", h)
    """
        w = [0.75, 0.5, 0.25, 1.06, 0.53]
        h = [0.75, 0.50, 0.25, 0.53, 1.06]
        这里的w和h都是相对于1*1坐标系的，
    """
    
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # print("stack后:", torch.stack((-w, -h, w, h)))
    # print("T后:", torch.stack((-w, -h, w, h)).T)    
    """
        stack后 = [[-0.75, -0.50, -0.25, -1.06, -0.53],
                    [-0.75, -0.50, -0.25, -0.53, -1.06],
                    [ 0.75,  0.50,  0.25,  1.06,  0.53],
                    [ 0.75,  0.50,  0.25,  0.53,  1.06]]
        T后 = [[-0.75, -0.75,  0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [-0.25, -0.25,  0.25,  0.25],
                [-1.06, -0.53,  1.06,  0.53],
                [-0.53, -1.06,  0.53,  1.06]]
        repeat应用到每个像素上，
        /2 获得从「中心」到「左上角」和「右下角」的偏移量，
    """


    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],    # 重复两次是因为要「左上」xy和「右下」xy
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)

    """
        shift_y = [0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875, 0.125, 0.375, 0.625, 0.875]
        shift_x = [0.125, 0.125, 0.125, 0.125, 0.375, 0.375, 0.375, 0.375, 0.625, 0.625, 0.625, 0.625, 0.875, 0.875, 0.875, 0.875]

        stack后 = [[0.12, 0.12, 0.12, 0.12],
                    [0.38, 0.12, 0.38, 0.12],
                    [0.62, 0.12, 0.62, 0.12],
                    [0.88, 0.12, 0.88, 0.12],
                    [0.12, 0.38, 0.12, 0.38],
                    [0.38, 0.38, 0.38, 0.38],
                    [0.62, 0.38, 0.62, 0.38],
                    [0.88, 0.38, 0.88, 0.38],
                    [0.12, 0.62, 0.12, 0.62],
                    [0.38, 0.62, 0.38, 0.62],
                    [0.62, 0.62, 0.62, 0.62],
                    [0.88, 0.62, 0.88, 0.62],
                    [0.12, 0.88, 0.12, 0.88],
                    [0.38, 0.88, 0.38, 0.88],
                    [0.62, 0.88, 0.62, 0.88],
                    [0.88, 0.88, 0.88, 0.88]]

        每一行代表一个锚框的「左上角」和「右下角」坐标，
        一组为一种锚框在整个图像上的位置，
        然后在这基础上复制boxes_per_pixel次，然后+偏移量

    """

    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)

    # 计算每个锚框的面积
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    # 计算交集
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # 取更靠近右下角的「左上角」坐标
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # 取更靠近左上角的「右下角」坐标
    # print(f"inter_upperlefts: {inter_upperlefts}, inter_lowerrights: {inter_lowerrights}")

    """
        inter_upperlefts,inter_lowerrights,inters的形状: (boxes1的数量,boxes2的数量,2)
        boxes1[:, None, :2] 将 boxes1的形状从(数量, 4)变为(数量, 1, 2)，
        boxes2[:, :2] 将 boxes2的形状从(数量, 4)变为(数量, 2)，
        这样可以进行广播操作，例如
        boxes1 = [[0, 0, 2, 2], [1, 1, 3, 3]] # AB
        boxes2 = [[2, 2, 4, 4], [4, 4, 6, 6]] # CD

        # 由于取max操作会触发「广播」机制，boxes1（数量, 1, 2）的第二维会与boxes2（数量, 2）的第二维进行广播，
        # 即 变为 （数量, 2, 2）

        boxes1[:, None, :2]:
            [
                [ [0, 0] ],  # 框A的左上角
                [ [1, 1] ]   # 框B的左上角
            ]
        广播第2维后得到:
            [
                [[0, 0], [0, 0]],  # 框A的左上角，一会分别跟框C和框D比较
                [[1, 1], [1, 1]]   # 框B的左上角，一会分别跟框C和框D比较
            ]

        boxes2[:, :2]:
            [
                [2, 2],  # 框C的左上角
                [4, 4]   # 框D的左上角
            ]
        广播第3维后得到:
            [
                [[2, 2], [4, 4]], # 框C的左上角，框D的左上角，用来跟A比较
                [[2, 2], [4, 4]]  # 框C的左上角，框D的左上角，用来跟B比较
            ]
        
        inter_upperlefts:
        逐元素取最大值后得到:
            [
                [[2, 2], [4, 4]], # A跟C比，C大； A跟D比，D大
                [[2, 2], [4, 4]]  # B跟C比，打平； B跟D比，D大
            ]
            
        inter_lowerrights:
        逐元素取最小值后得到:
            [
                [[2, 2], [2, 2]], # A跟C比，A小； A跟D比，A小
                [[3, 3], [3, 3]]  # B跟C比，打平； B跟D比，B小
            ]



    """

    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) # 将负值置为0
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)

    inter_areas = inters[:, :, 0] * inters[:, :, 1] # 计算交集面积
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

# ======================================== 偏移量转换 ========================================

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

# ======================================== 锚框标记 ========================================

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""

    # ground_truth的形状是(num_gt_boxes, 4)，anchors的形状是(num_anchors, 4)
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    
    jaccard = box_iou(anchors, ground_truth)
    # jaccard形状：(num_anchors, num_gt_boxes)
    # jaccard[i,j]表示锚框i与真实框j的 IoU

    # ------ 先处理满足IoU阈值的锚框 ------
    
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device) # 初始化为-1，表示未分配
    
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1) # 每个「锚框」对应的最大IoU和索引（每行的最大值和索引）

    # print(f"max_ious: {max_ious}, indices: {indices}")

    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 找到满足IoU阈值的锚框索引 行
    box_j = indices[max_ious >= iou_threshold]                   # 找到对应的真实边界框索引  列

    # print(f"anc_i: {anc_i}, box_j: {box_j}")
    anchors_bbox_map[anc_i] = box_j
    print(f"anchors_bbox_map: {anchors_bbox_map}")

    """
        此时已经完成了从「锚框」出发的分配，锚框对应的、符合条件的真实边界框的索引已经存储在anchors_bbox_map中。
        其实这两部分的分配会有重叠，我们取并集即可。
    """

    # ------ 处理没有满足IoU阈值的锚框 ------

    # 如果没有满足IoU阈值的锚框，则将其分配给与其IoU最大的真实边界框（确保每个真实框至少有一个锚框）
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long() # 列索引（对应真实边界框）
        anc_idx = (max_idx / num_gt_boxes).long() # 行索引（对应锚框）
        anchors_bbox_map[anc_idx] = box_idx
        print(f"分配锚框 {anc_idx} 给真实边界框 {box_idx}")
        print(f"anchors_bbox_map: {anchors_bbox_map}")

        jaccard[:, box_idx] = col_discard # 删除该列
        jaccard[anc_idx, :] = row_discard # 删除该行
    return anchors_bbox_map

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""

    # anchors的形状是(batch_size, num_anchors, 4)，labels的形状是(batch_size, num_gt_boxes, 5)  5 = 类别 + 4个坐标

    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]


    for i in range(batch_size):

        label = labels[i, :, :] # 取出第i个batch的标签（num_gt_boxes, 5）
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device) # label[:, 1:] 取出四个坐标
        
        # 此时anchors_bbox_map的形状是(num_anchors,)，索引为锚框，值为分配的真实边界框索引（每一个「真实边界框」都至少分配给一个「锚框」）
        
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

        """
            假如anchors_bbox_map是[-1, 0, 1, -1, 1]
            bbox_mask是[[0., 0., 0., 0.],
                        [1., 1., 1., 1.],
                        [1., 1., 1., 1.],
                        [0., 0., 0., 0.],
                        [1., 1., 1., 1.]]
        """

        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        print(f"indices_true: {indices_true}, bb_idx: {bb_idx}")
        """
            假设anchors_bbox_map = [-1, 0, 1, -1, 1]
            indices_true = [[1], [2], [4]] 
            bb_idx = [0, 1, 1]
        """

        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # 分配「真实边界框」的类别标签
        assigned_bb[indices_true] = label[bb_idx, 1:]            # 分配「真实边界框」的坐标
        print(f"class_labels: {class_labels}, assigned_bb: {assigned_bb}")
        """
            class_labels: [0, 1, 2, 0, 2]
            assigned_bb: [[0.00, 0.00, 0.00, 0.00],
                            [0.10, 0.08, 0.52, 0.92],
                            [0.55, 0.20, 0.90, 0.88],
                            [0.00, 0.00, 0.00, 0.00],
                            [0.55, 0.20, 0.90, 0.88]]
        """

        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask # 忽略 未分配的锚框，计算「锚框」和「真实边界框」之间的偏移量
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


# ======================================== 非极大值抑制 ========================================

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    # boxes的形状是(num_boxes, 4)，为每个预测边界框的坐标
    # scores的形状是(num_boxes,)，为每个预测边界框的置信度
    # iou_threshold是一个阈值，用于非极大值抑制

    B = torch.argsort(scores, dim=-1, descending=True) # 此时B中为按置信度排序的「边界框索引」

    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:

        # print(f"当前B: {B}")
        # 取置信度最高的一个
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break # 如果只剩下一个边界框，则直接保留

        # 计算当前边界框与其他边界框的IoU
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1) 
        
        # 如果IoU小于阈值，则保留该边界框
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # print(f"保留索引: {inds}")

        # 更新保留的边界框索引
        B = B[inds + 1] # 这里要加回来1，因为前面对比iou时已经去掉了第一个元素
        # print(f"更新后的B: {B}")
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""

    # cls_probs的形状是(batch_size, num_classes, num_anchors) 每个「锚框」的类别概率
    # offset_preds的形状是(batch_size, num_anchors, 4) 每个「锚框」的偏移量预测
    # anchors的形状是(batch_size, num_anchors, 4) 每个「锚框」的坐标

    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]

    out = []
    for i in range(batch_size):

        # 取出预测的类别概率和偏移量
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0) # 取出每个锚框的最大类别概率（置信度）和对应的类别索引，除了背景类（索引0）

        # 将偏移量应用到锚框上
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

# ======================================== 边界框显示 ========================================

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))







if __name__ == "__main__":
    # 读取图片
    img = plt.imread('DiveInDL/img/catdog.jpg')
    print(img.shape)  # (h, w, c) h:高度，w:宽度，c:通道数
    h, w = img.shape[:2]

    # ======================================== 测试锚框生成 ========================================
    X = torch.rand(size=(1, 1, 4, 4))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)  # (batch_size, num_anchors, 4)

    X = torch.rand(size=(1, 3, h, w)) # 生成一个随机的输入张量（batch_size=1, 通道数=3, 高度=h, 宽度=w）
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)  # (batch_size, num_anchors, 4) 4: (xmin, ymin, xmax, ymax)

    boxes = Y.reshape(h, w, 5, 4) # 将锚框张量重塑为(h, w, boxes_per_pixel, 4)的形状
    print(boxes[250, 250, 0, :]) # 访问以（250,250）为中心的第一个锚框

    # 绘制锚框
    bbox_scale = torch.tensor([w, h, w, h], device=Y.device)  # 缩放因子
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)

    show_bboxes(ax, boxes[250, 250] * bbox_scale, 
                labels=['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
    # plt.show()

    # ======================================== 测试锚框标记 ========================================
    boxes1 = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]])
    boxes2 = torch.tensor([[2, 2, 4, 4], [4, 4, 6, 6]])
    iou = box_iou(boxes1, boxes2)
    print("交并比：", iou)

    # 测试锚框认领
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    show_bboxes(ax, boxes[250, 250] * bbox_scale,
                labels=['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
                        's=0.75, r=0.5'])
    show_bboxes(ax, ground_truth[:, 1:] * bbox_scale, labels=['cat', 'dog'])

    labels = multibox_target(anchors.unsqueeze(0), ground_truth.unsqueeze(0))
    print("偏移量:", labels[0])
    print("掩码:", labels[1])
    print("类别标签:", labels[2])

    # ========================================= 测试非极大值抑制 =========================================
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                        [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                        [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    show_bboxes(ax, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    plt.show()  # 显示图像和边界框
    
    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
    print("输出:", output)  

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(ax, [torch.tensor(i[2:]) * bbox_scale], label)

    plt.show()  # 显示图像和边界框