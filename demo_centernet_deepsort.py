from pprint import pprint
import numpy as np
import cv2
import os


# CenterNet
import sys
CENTERNET_PATH = './CenterNet/src/lib'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts

MODEL_PATH = './CenterNet/models/ctdet_coco_dla_2x.pth'
ARCH = 'dla_34'

# MODEL_PATH = './CenterNet/models/ctdet_coco_resdcn18.pth'
# ARCH = 'resdcn_18'

# # MODEL_PATH = './CenterNet/models/model_best.pth'
# # ARCH = 'dla_34'

# MODEL_PATH = './CenterNet/models/model_best2129_resdcn18.pth'
# ARCH = 'resdcn_18'


TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {} --arch {}'.format(TASK, MODEL_PATH, ARCH).split(' '))

# vis_thresh  这里修改置信度阈值
opt.vis_thresh = 0.35

"""
coco_class = [
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
"""

# 分类号 参考上面class序号进行过滤，如person是1 car是3 如果不需要过滤 请注释该变量，如果是你自己的model，请按照自己的编号进行
# specified_class_id_filter = 1


# input_type
# for video, 'vid',  for webcam, 'webcam', for ip camera, 'ipcam'
opt.input_type = 'vid'

# ------------------------------
# # for video
opt.vid_path = './vehicle.mp4'  #
# for video
# opt.vid_path = '/home/joe/Documents/playground/centerNet-deep-sort/337.mp4'  #
# ------------------------------
# for webcam  (webcam device index is required)
opt.webcam_ind = 0
# ------------------------------
# for ipcamera (camera url is required.this is dahua url format)
opt.ipcam_url = 'rtsp://{0}:{1}@IPAddress:554/cam/realmonitor?channel={2}&subtype=1'
# ipcamera camera number
opt.ipcam_no = 8
# ------------------------------


# from util import COLORS_10, add_cls_confi_draw_bboxes
from deep_sort import DeepSort
from util import *
import time
# 待修改


def bbox_to_xywh_cls_conf(bbox):
    """flat nesting results.

    Args:
      bbox:  structure        {1:array([[ 4.8216104e+02,  5.2652222e+02,  6.7258179e+02,  1.0477332e+03, 7.4529582e-01],...],
                              2:array([[ 4.8216104e+02,  5.2652222e+02,  6.7258179e+02,  1.0477332e+03, 7.4529582e-01],...],
                              x:.....}
    Returns:
      x1x2y1y2, confidence, class_name
    """

    # bbox 是字典，键为int类别，值为bbox array([x1,y1,x2,y2,confidence],...)
    # all class
    new_bbox = []  # 去除 序号为0的背景在字典里已经去除了，平铺所有class的数据（去一层括号，拼接)
    for cls_num, box in bbox.items():
        if not box.any():
            pass
        else:
            for single_box in box:
                if not single_box.any():
                    pass
                else:
                    # print("xxxxx",single_box)
                    a = np.append(single_box, cls_num)
                    new_bbox.append(a)
    new_bbox = np.array(new_bbox)
    # print("asdasd",new_bbox)
    # new_box 结构 [
    #              [x1,y1,x2,y2,confidence,cls_num],
    #              [x1,y1,x2,y2,confidence,cls_num],
    #               ...]

    if any(new_bbox[:, 4] > opt.vis_thresh):

        # 第五位是 confidence 这里过滤不合格的部分
        new_bbox = new_bbox[new_bbox[:, 4] > opt.vis_thresh, :]
        new_bbox[:, 2] = new_bbox[:, 2] - new_bbox[:, 0]  # x2 变成 w
        new_bbox[:, 3] = new_bbox[:, 3] - new_bbox[:, 1]  # y2 变成 h

        return new_bbox[:, :4], new_bbox[:, 4], new_bbox[:, 5]
        # return [[x,y,w,h], ...], [confidence,...], [cls_num,...]
    else:

        return None, None, None


class Detector(object):
    def __init__(self, opt):
        self.vdo = cv2.VideoCapture()

        # centerNet detector
        self.detector = detector_factory[opt.task](opt)
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")

        self.write_video = True

    def open(self, video_path):

        if opt.input_type == 'webcam':
            self.vdo.open(opt.webcam_ind)

        elif opt.input_type == 'ipcam':
            # load cam key, secret
            with open("cam_secret.txt") as f:
                lines = f.readlines()
                key = lines[0].strip()
                secret = lines[1].strip()

            self.vdo.open(opt.ipcam_url.format(key, secret, opt.ipcam_no))

        # video
        else:
            assert os.path.isfile(opt.vid_path), "Error: path error"
            self.vdo.open(opt.vid_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:

            # 原版AVI格式
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # self.output = cv2.VideoWriter("demo1.avi", fourcc, 20, (self.im_width, self.im_height))

            # MP4 格式
            encode = cv2.VideoWriter_fourcc(*'mp4v')
            # 正式版本用
            # self.output = cv2.VideoWriter("./output/demo_{}.mp4".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ), encode, 24, (self.im_width, self.im_height))
            # 调试版本
            # # self.output = cv2.VideoWriter("./output/demo.mp4", encode, 24, (self.im_width, self.im_height))
            self.output = cv2.VideoWriter(
                "./output.mp4", encode, 24, (self.im_width, self.im_height))
            # self.output = cv2.VideoWriter("./output/demo.mp4", encode, 24, (self.im_width, self.im_height))

        # return self.vdo.isOpened()

    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        #avg_fps_sum = 0.0
        #time_cost = 0
        while self.vdo.grab():

            frame_no += 1
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]
            #im = ori_im[ymin:ymax, xmin:xmax, :]
            image_h, image_w, _ = im.shape
            start = time.time()

            results = self.detector.run(im)['results']

            ####################### 筛选要跟踪的指定分类号的目标 ########################
            try:
                results = dict((key, value) for key, value in results.items() if key == specified_class_id_filter)
            except NameError:
                pass
            else:
                pass

            # results          {1:array([[ 4.8216104e+02,  5.2652222e+02,  6.7258179e+02,  1.0477332e+03, 7.4529582e-01],...],
            #                  2:array([[ 4.8216104e+02,  5.2652222e+02,  6.7258179e+02,  1.0477332e+03, 7.4529582e-01],...],
            #                  x:.....}
            # 1,2,...x == class_num
            # array[0:4] == xywh
            # array[4] == confidence
            bbox_xywh, cls_conf, cls_num = bbox_to_xywh_cls_conf(results)
            # bbox_xywh [[1711.6794    575.22345    40.046265   93.066284],[1443.3882    454.66113    32.5177     99.92456 ]]
            # cls_conf  [0.59678245 0.5107993 ]

            # 绘制计数线和危险区域
            im = draw_line_and_area(im, image_h, image_w)

            if bbox_xywh is not None:
                # points =[[跟踪id对应的双向队列],.....] e.g.points[1] 是 跟踪号为1的轨迹点历史记录
                outputs, points = self.deepsort.update(
                    bbox_xywh, cls_conf, cls_num, im)
                # [x1,y1,x2,y2,track_id,confidences,cls_num]

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]   # [[x1,y1,x2,y2],....]
                    identities = outputs[:, 4]  # track_id [1,2]
                    confidences = outputs[:, 5]  # confidences [54,62]
                    class_nums = outputs[:, -1]  # class_num [1,1]
                    # print("track_id {} confidences {}".format(identities,confidence))
                    # ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))
                    ori_im = add_cls_confi_draw_bboxes(
                        ori_im, bbox_xyxy, identities, confidences, class_nums, points, offset=(xmin, ymin))

            end = time.time()
            time_gap = end - start
            # FPS相关信息
            fps = 1 / time_gap
            ###################### TIME COST LOG 记录#################################
            # print("centernet_res18_timecost {} {}\n".format(frame_no, 1/fps )) # python demo_centernet_deepsort.py | grep centernet_timecost >  centernet.log

            # 平均fps
            # avg_fps_sum += fps
            # avg_fps = avg_fps_sum / frame_no

            # time_cost += end - start
            # 绘制面板及计数文字信息
            ori_im = draw_data_panel(ori_im, bbox_xywh, fps)

            #print("deep time: {}s, fps: {}".format(end - start_deep_sort, 1 / (end - start_deep_sort)))

            # #print("centernet time: {:.3f}s, fps: {:.3f}, avg fps : {:.3f}".format(end - start, fps,  avg_fps_sum/frame_no))

            # cv2.putText(ori_im, "Current FPS: {:.3f} Average_FPS: {:.3f} @Laptop GTX1060".format(fps , avg_fps_sum/frame_no ), (image_w//3, 20), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            # cv2.putText(ori_im, "Current Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ), (image_w//3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            # cv2.putText(ori_im, "Monitoring Duration: {}".format(time.strftime("%H:%M:%S", time.gmtime(time_cost)) ), (image_w//3, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            # cv2.putText(ori_im, "@Author: {}".format("Clemente"), (image_w//3, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            cv2.imshow("Traffic Flow Counter", ori_im)
            cv2.waitKey(1)

            if self.write_video:
                self.output.write(ori_im)


if __name__ == "__main__":
    import sys

    # if len(sys.argv) == 1:
    #     print("Usage: python demo_yolo3_deepsort.py [YOUR_VIDEO_PATH]")
    # else:
    cv2.namedWindow("Traffic Flow Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Flow Counter", 1080, 720)

    #opt = opts().init()
    det = Detector(opt)

    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open("MOT16-11.mp4")
    det.detect()
