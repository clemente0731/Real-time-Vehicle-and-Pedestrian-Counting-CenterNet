import numpy as np
import cv2
import random
import colorsys
import collections
import time
# from playsound import playsound
from pprint import pprint


# # 记录目标对应的计数
# OrderedDict([('river_boat', {'down': 0, 'left': 0, 'right': 0, 'up': 0}),
#              ('speedboat', {'down': 0, 'left': 1, 'right': 0, 'up': 0})])
counter_dict = collections.OrderedDict()

# # 计数记忆(保证不重复计数)range的参数根据你的视频目标数保持数量级一致
counter_memory = dict.fromkeys(range(54000), 0)

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

### 你的面板类别打印设置
class_names = [
      '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#### 是否开启警戒区检测
#warning_area_monitoring_switch = True
warning_area_monitoring_switch = False

### 警戒区域 从左到右 顺时针点位
if warning_area_monitoring_switch:
    # 警戒区点坐标 （需要与直线斜率一致）
    polys = np.array([[800, 150],[1100, 150],[1050, 230],[750, 230],], np.int32)
    poly_points = polys.tolist()
    polys = polys.reshape((-1,1,2))

else:
    pass
### 字体样式
font_style = cv2.FONT_HERSHEY_SIMPLEX

frame_num =0 

############### 直线两个点设置##################

# 近似垂直线
# line = [(680, 100), (265, 720)]
# line = [(812, 89), (692, 1078)]
# line = [(748, 218), (656, 756)]
# line = [(920, 0), (920, 1080)] # HPJ001
# line = [(623, 0), (623, 1080)] # HPJ002

#line = [(922, 0), (922, 1200)] # WT002 WT001


# # 近似水平线
# line = [(583, 839), (1906, 652)] # WSK001X24
# line = [(0, 600), (1920, 600)] # HZNH1
# line = [(0, 830), (1920, 830)] # HZNH2
# line = [(0, 530), (2100, 530)] # for vehicle.mp4
line = [(0, 430), (2100, 430)] # for people.mp4

##### 近似计算直线 垂直 还是水平
def assess_horizontal_or_vertical(line):
    hrz_difference = line[1][0]-line[0][0]
    vtc_difference = line[1][1]-line[0][1]
    squared_difference =  hrz_difference ** 2 - vtc_difference ** 2
    if squared_difference >= 0:
        horizontal_true_vertical_false = True
    else:
         horizontal_true_vertical_false = False

    return  horizontal_true_vertical_false

##### 当前计数线垂直还是水平，如果情况特殊可以手动设值，horizontal(True) or vertical(False)
horizontal_True_vertical_False = assess_horizontal_or_vertical(line)



def draw_data_panel(img, bboxes, fps):
    """Draw the panel.

    Args:
      img:              image
      bbox:             bbox
      identities:       identities
      offset:           offset

    Returns:
      The new image.
    """

    if bboxes is None:
        target_num = 0
    else:
        target_num =  len(bboxes)

    global counter_dict
    global font_style
    num_recorded_class = len(counter_dict)

    # # 图层1 先绘制信息面板矩形，以保持透明底层
    # alpha = 0.3
    # image_h, image_w, _ = img.shape
    # overlay = img.copy() # img副本 以供填充覆盖
    # cv2.rectangle(img, (0,0), (image_w//3 ,num_recorded_class * 40 + 70), (32,36,46), thickness=-1)
    # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0) # img副本叠加

    # 图层1 先绘制信息面板矩形，以保持透明底层
    alpha = 0.3
    image_h, image_w, _ = img.shape
    overlay = img.copy() # img副本 以供填充覆盖
    cv2.rectangle(img, (0,0), (image_w//3 -70 ,num_recorded_class * 40 + 100), (32,36,46), thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0) # img副本叠加


    # 图层2 信息面板文字描述

    vertical_increment = 20
    vertical_correction = 20
    horizontal_increment = image_w // 5
    up_or_left_sum = 0
    down_or_right_sum = 0
    text_thickness = int(0.6 * (image_h + image_w) / 1000)
    font_scale = 0.5
    # sum 纵向坐标偏移量
    sum_increment = num_recorded_class * 20 +40

    ################ 按目标分类的计数信息填入 ####################
    # counter_dict = { speedboat {'up': 0, 'down': 0, 'left': 0, 'right': 0}, ..., river_boat {'up': 0, 'down': 0, 'left': 0, 'right': 0} }
    for key,values in counter_dict.items():
        vertical_correction += vertical_increment # 每次新的键值对 纵向坐标 ++vertical_increment
        # 检测物类别具体描述 type
        cv2.putText(img, " {}".format(key) ,(0, vertical_correction), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)

        # 计数线接近水平放置时
        if horizontal_True_vertical_False:
            # up内河计数
            cv2.putText(img,"{}".format(values['up']),(horizontal_increment, vertical_correction), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
            # down内河计数
            cv2.putText(img,"{}".format(values['down']),(horizontal_increment+50, vertical_correction), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
            # 上下 累计计数
            up_or_left_sum += values['up']
            down_or_right_sum += values['down']

        # 计数线近似垂直放置时
        else:
            # left内河计数
            cv2.putText(img,"{}".format(values['left']),(horizontal_increment, vertical_correction), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
            # right内河计数
            cv2.putText(img,"{}".format(values['right']),(horizontal_increment+50, vertical_correction), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
            # 左右 累计计数
            up_or_left_sum += values['left']
            down_or_right_sum += values['right']



    ################ 不分左右上下的计数信息面板 ####################
    # 左下角 累计值
    cv2.putText(img, " cumulative count" ,(0, sum_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
    # 左下角 当前值 current_targets
    cv2.putText(img, " target number" ,(0, sum_increment+20), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
    # 左下角 FPS 当前值
    cv2.putText(img, " fps" ,(0, sum_increment+40), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
    # # 左下角 时间
    # cv2.putText(img, " time" ,(0, sum_increment+60), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
    # 左下角 主网络
    cv2.putText(img, " detector" ,(0, sum_increment+60), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
    # 左下角 署名
    cv2.putText(img, " author" ,(0, sum_increment+80), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)


    # 左上角 分类计数
    cv2.putText(img," type",(0, vertical_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)

    if horizontal_True_vertical_False:
        # up_count
        cv2.putText(img,"up",(horizontal_increment, vertical_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # down_count
        cv2.putText(img,"down",(horizontal_increment+50, vertical_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 目前up/down计数的 cumulative count 数字
        cv2.putText(img,"{}".format(up_or_left_sum),(horizontal_increment, sum_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{}".format(down_or_right_sum),(horizontal_increment+50, sum_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 当前目标数 current_targets
        cv2.putText(img,"{}".format(target_num),(horizontal_increment , sum_increment + 20 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{}".format(target_num),(horizontal_increment + 50, sum_increment + 20 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 当前FPS资料
        cv2.putText(img,"{:0.1f}".format(fps),(horizontal_increment , sum_increment + 40 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{:0.1f}".format(fps),(horizontal_increment + 50, sum_increment + 40), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # # 当前时间
        # cv2.putText(img,"{}".format( time.strftime("%Y%m%d %H:%M:%S", time.localtime()) ),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 主网络
        cv2.putText(img,"{}".format("CenterNet"),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 署名描述
        cv2.putText(img,"{}".format("Clemente420"),(horizontal_increment , sum_increment + 80 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)

    else:
        # left_count
        cv2.putText(img,"left",(horizontal_increment, vertical_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # right_count
        cv2.putText(img,"right",(horizontal_increment+50, vertical_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 目前left/right计数的 cumulative count 数字
        cv2.putText(img,"{}".format(up_or_left_sum),(horizontal_increment, sum_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{}".format(down_or_right_sum),(horizontal_increment+50, sum_increment), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 当前目标数 current_targets
        cv2.putText(img,"{}".format(target_num),(horizontal_increment , sum_increment + 20 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{}".format(target_num),(horizontal_increment + 50, sum_increment + 20 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 当前FPS资料
        cv2.putText(img,"{:0.1f}".format(fps),(horizontal_increment , sum_increment + 40 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img,"{:0.1f}".format(fps),(horizontal_increment + 50, sum_increment + 40), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # # 当前时间
        # cv2.putText(img,"{}".format( time.strftime("%Y%m%d %H:%M:%S", time.localtime()) ),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 主网络
        cv2.putText(img,"{}".format("CenterNet"),(horizontal_increment , sum_increment + 60 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)
        # 署名描述
        cv2.putText(img,"{}".format("Clemente420"),(horizontal_increment , sum_increment + 80 ), font_style, font_scale, (8,196,254), thickness = text_thickness, lineType=cv2.LINE_AA)

    ################## 记录当前检测到的目标数量 #####################
    # global frame_num
    # frame_num += 1
    # print("target_num {} {}\n".format(frame_num, target_num )) 
    ########### 不分左右上下的计数信息面板 #######################

    # FPS当前
    # 时间
    # 署名作者
    return img

def draw_line_and_area(img,image_h, image_w,line=line):
    """Draw the panel.

    Args:
      img:              image
      bbox:             bbox
      identities:       identities
      offset:           offset

    Returns:
      The new image.
    """
    #############################################################
    # 画警戒区　
    if warning_area_monitoring_switch:
        cv2.polylines(img, [polys], isClosed=True, color=(0,0,255), thickness=1, lineType=cv2.LINE_AA)
    else:
        pass

    #############################################################
    # 判断水平或垂直计数，然后draw line
    if horizontal_True_vertical_False:
        cv2.line(img, line[0], line[1], (8,196,254), thickness=2, lineType=cv2.LINE_AA)  # 参数要求整数 除法用//
    else:
        cv2.line(img, line[0], line[1], (8,196,254), thickness=2, lineType=cv2.LINE_AA)  # 参数要求整数 除法用//

    return img


# 判断是否进入警戒区
def is_point_in(x, y, polygon_points):
    count = 0
    x1, y1 = polygon_points[0]
    x1_part = (y1 > y) or ((x1 - x > 0) and (y1 == y)) # x1在哪一部分中
    x2, y2 = '', ''  # points[1]
    polygon_points.append((x1, y1))
    for point in polygon_points[1:]:
        x2, y2 = point
        x2_part = (y2 > y) or ((x2 > x) and (y2 == y)) # x2在哪一部分中
        if x2_part == x1_part:
            x1, y1 = x2, y2
            continue
        mul = (x1 - x)*(y2 - y) - (x2 - x)*(y1 - y)
        if mul > 0:  # 叉积大于0 逆时针
            count += 1
        elif mul < 0:
            count -= 1
        x1, y1 = x2, y2
        x1_part = x2_part
    if count == 2 or count == -2:
        return True
    else:
        return False

def intersect(A,B,C,D): # ab是目标框中心的两个近点，cd是线段
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])



def draw_bbox(img, box, cls_name, identity=None, offset=(0,0)):
    '''
        draw box of an id
    '''
    x1,y1,x2,y2 = [int(i+offset[idx%2]) for idx,i in enumerate(box)]
    # set color and label text
    color = COLORS_10[identity%len(COLORS_10)] if identity is not None else COLORS_10[0]
    label = '{} {}'.format(cls_name, identity)
    # box text and bar
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1),(x2,y2),color,2)
    cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 1)
    return img


def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    """Draw the bboxes.

    Args:
      img:              image
      bbox:             bbox
      identities:       identities
      offset:           offset

    Returns:
      The new image.
    """
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id%len(COLORS_10)]
        label = '{} {}'.format("object", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

def add_cls_confi_draw_bboxes(img, bbox, identities=None, confidences=None, class_nums=None,points=None, offset=(0,0)):
    image_h, image_w, _ = img.shape
    num_classes = len(class_names)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    # 颜色是反的　需要从BGR 转换到　RGB (0,1,2) 需要转换到　(2,1,0)
    p0 = (0, 0)
    p1 = (0, 0)

    global font_style



    for i,box in enumerate(bbox):
        # 检测框的左上和右下两个点坐标
        x1,y1,x2,y2 = [int(i) for i in box]

        # 偏移量修正
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # 跟踪识别号
        track_id = int(identities[i]) if identities is not None else 0

        # 该目标的分类名
        class_name = class_names[class_nums[i]]
        # 该目标的置信度
        confidence = confidences[i]/100

        # 检测框的颜色
        bbox_color = colors[class_nums[i]]
        # 检测框的粗细值
        bbox_thick = int(0.7 * (image_h + image_w) / 500)
        fontScale = 0.5 # 分类字体和填充框大小

        #############################################################
        # 对应目标的计数字典
        global counter_dict
        # 如果不存在相应分类的键，则初始化键值对为0
        if class_name not in counter_dict:
            counter_dict[class_name] = {}
            counter_dict[class_name]['up'] =0
            counter_dict[class_name]['down'] =0
            counter_dict[class_name]['left'] =0
            counter_dict[class_name]['right'] =0
        else:
            pass


        #############################################################
        # 统计部分
        if len(points[track_id]) >= 3:
            p0 = points[track_id][-1] # p0是最新的目标点中心
            p1 = points[track_id][-3] # p1 是前四帧的目标点中心

            # 判断进出 p0是当前帧目标中心坐标，p1是前一帧的目标中心点数
            if intersect(p0, p1, line[0], line[1]) and counter_memory[track_id] != 1:
                # 如果是横向统计
                if horizontal_True_vertical_False:
                    if p0[1] < p1[1]: #最新点的y坐标小于 之前点的y坐标 在向上走
                        counter_dict[class_name]['up'] += 1   #字典是嵌套形式
                        counter_memory[track_id] = 1
                    elif p0[1] > p1[1]: #最新点的y坐标 大于 之前点的y坐标 在向下走
                        counter_dict[class_name]['down'] += 1
                        counter_memory[track_id] = 1
                    else:
                        pass
                # 如果是左右统计
                else:
                    if p0[0] < p1[0]: #最新点的x坐标小于 之前点的x坐标 在向左走
                        counter_dict[class_name]['left'] += 1   #字典是嵌套形式
                        counter_memory[track_id] = 1
                    elif p0[0] > p1[0]: #最新点的x坐标 大于 之前点的x坐标 在向右走
                        counter_dict[class_name]['right'] += 1
                        counter_memory[track_id] = 1
                    else:
                        pass

        #############################################################
        # 是否开启警戒区检测
        if warning_area_monitoring_switch:
            # 进入警戒区
            if is_point_in(p0[0], p0[1], poly_points):

                # 报警声音 TODO

                # 警戒区 显示WARNING
                # cv2.putText(img,"WARNING!!!",(p0[0] - 30, p0[1] + 20), font_style, 0.6, (0,0,0), 1,lineType=cv2.LINE_AA)
                # cv2.putText(img,"RESTRICTED AREA",(p0[0] - 70, p0[1] + 40), font_style, 0.6, (0,0,0), 1,lineType=cv2.LINE_AA)
                cv2.arrowedLine(img, (p0[0]-80, p0[1]-10), (p0[0]-40, p0[1]-10), (0, 0, 0), thickness=2, line_type=cv2.LINE_AA, shift=0, tipLength=0.2)
                cv2.arrowedLine(img, (p0[0]+80, p0[1]-10), (p0[0]+40, p0[1]-10), (0, 0, 0), thickness=2, line_type=cv2.LINE_AA, shift=0, tipLength=0.2)
                cv2.putText(img,"RESTRICTED WARNING!",(771, 296), font_style, 1, (31,31,197), 2,lineType=cv2.LINE_AA)
                cv2.rectangle(img,(746, 262),(1135 ,320),(31,31,197),3)
                # 内部填充
                overlay = img.copy() # img副本 以供填充覆盖
                cv2.fillPoly(overlay, [polys], color=(0,0,255))
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0) # img副本叠加

            # 警戒区以外的区域
            else:
                pass
        # 不开启警戒区检测
        else:
            pass


        #######################################################
        #####TODO 安全距离警告 #########################
        # cv2.putText(img,"DISTANCE WARNING!",(705, 515), font_style, 1.5, (31,31,197), 3,lineType=cv2.LINE_AA)
        # cv2.rectangle(img,(675, 432),(1200,574),(31,31,197),5)

        #############################################################
        # 检测框及相关属性描述
        # 检测框
        cv2.rectangle(img,(x1, y1),(x2,y2), bbox_color, bbox_thick)
        # 描述标记文字和文字框
        label = "{}: {}".format(class_name, confidence)
        t_size = cv2.getTextSize(label, 0, fontScale, thickness=bbox_thick)[0]
        # 画分类处的文字框
        cv2.rectangle(img, (x1, y1 -3), (x1 + t_size[0], y1 - t_size[1] - 6), bbox_color, thickness=-1) # 填充
        cv2.putText(img,label,(x1,y1 - 5), font_style, fontScale, (0,0,0), bbox_thick//3,lineType=cv2.LINE_AA)




        ###############################################
        # 画轨迹　序号就是　对应的跟踪id号　point=[[跟踪id对应的双向队列],.....] 包裹队列的列表
        for j in range(1, len(points[track_id])):
            if points[track_id][j - 1] is None or points[track_id][j] is None:
                ###############################################
                # TODO 滑动平均处理历史轨迹点  moving_average
                # points[track_id] = points[track_id] # 滑动平均处理 moving_average
                continue
            # thickness = int(np.sqrt(32 / float(j + 1)) * 2) #第一个点重　后续线逐渐变细
            # 轨迹曲线绘制
            cv2.line(img,(points[track_id][j-1]), (points[track_id][j]),(8,196,255),thickness = 2,lineType=cv2.LINE_AA)

        # 链接两个点为直线（跟踪点）
        # cv2.line(img, p0, p1, (8,196,254), 5,lineType=cv2.LINE_AA)  # 把这两个中心的点连接

        ## 跟踪的序列号
        # cv2.putText(img,"{}".format(track_id),(p0[0]+5, p0[1]+5), font_style, 0.8*fontScale, (32,36,46), 1,lineType=cv2.LINE_AA)
        # 画圆心
        # cv2.circle(img,  (p0), radius=3, color=(46, 36, 32), thickness=-1,lineType=cv2.LINE_AA)


        # 跟踪标记小圆点
        cv2.circle(img,  (p0), radius=3, color=(46, 36, 32), thickness=-1,lineType=cv2.LINE_AA)
        # 跟踪识别号
        cv2.putText(img,"{}".format(track_id),(p0[0]+5, p0[1]+5), font_style, 0.8*fontScale, (32,36,46), 1,lineType=cv2.LINE_AA)



    # # 旧版经典三原色
    # # 分类计数面板矩形 先绘制 后绘制文字
    # cv2.rectangle(img, (0,0), (horizontal_increment,image_h//15 * 5+20), (8,196,254), thickness=-1)
    # # up/left 内河计数面板矩形
    # cv2.rectangle(img, (horizontal_increment,0), (horizontal_increment+50,image_h//15 * 5+20), (110,90,208), thickness=-1)
    # # down/right 内河计数面板矩形
    # cv2.rectangle(img, (horizontal_increment+50,0), (horizontal_increment+100,image_h//15 * 5+20), (211,0,148), thickness=-1)

    return img


def softmax(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(x*5)
    return x_exp/x_exp.sum()

def softmin(x):
    assert isinstance(x, np.ndarray), "expect x be a numpy array"
    x_exp = np.exp(-x)
    return x_exp/x_exp.sum()



if __name__ == '__main__':
    x = np.arange(10)/10.
    x = np.array([0.5,0.5,0.5,0.6,1.])
    y = softmax(x)
    z = softmin(x)
    import ipdb; ipdb.set_trace()