import numpy as np

from deep.feature_extractor import Extractor
from sort.nn_matching import NearestNeighborDistanceMetric
from sort.preprocessing import non_max_suppression
from sort.detection import Detection
from sort.tracker import Tracker
import cv2
import time
from collections import deque
# 记住历史中心点，用作画轨迹  maxlen轨迹线长度
# 序号就是　对应的跟踪id号　points=[[跟踪id对应的双向队列],.....] 包裹队列的列表
points = [deque(maxlen=5) for _ in range(5400)]
# points = [ 某跟踪识别号的deque([(1193, 203),(1192, 203),(1190, 203),maxlen=10), ....]


class DeepSort(object):
    def __init__(self, model_path):
        # 检测结果阈值。低于这个阈值的检测结果将会被忽略  # 过滤掉置信度小于self.min_confidence的bbox，生成detections
        self.min_confidence = 0.25
        self.nms_max_overlap = 1.0  # 非极大抑制的阈值 原始值1.0
        # NMS (这里self.nms_max_overlap的值为1，即保留了所有的detections)
        self.extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = 0.2  # 0.2 余弦距离的控制阈值 调节这个能改善IDsw
        # 描述的区域的最大值 它是一个列表，列出了每次出现曲目的特征。nn_bodget确定此列表的大小。例如，如果它是10，则仅存储曲目在板上出现的最后10次的特征
        nn_budget = 100
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xywh, confidences, class_num, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        detections = []
        try:
            features = self._get_features(bbox_xywh, ori_img)
            for i, conf in enumerate(confidences):
                if conf >= self.min_confidence and features.any():
                    # Detection 在detection.py找到相关的类
                    detections.append(
                        Detection(bbox_xywh[i], conf, class_num[i], features[i]))
                else:
                    pass
        except Exception as ex:
            # TODO Error: OpenCV(4.1.1) /io/opencv/modules/imgproc/src/resize.cpp:3720: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
            print("{} Error: {}".format(time.strftime(
                "%H:%M:%S", time.localtime()), ex))
            # print('Error or video finish ')

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(
            boxes, self.nms_max_overlap, scores)  # indices = [0] 或者 [0,1]
        detections = [detections[i]
                      for i in indices]  # 根据编号 做 嵌套的list[ [0编号],[1编号] ]
        # print(detections[0].confidence)
        # confidence: 0.5057685971260071
        # print(detections)
        # [bbox_xywh: [1508.47619629  483.33926392   34.95910645   77.69906616],
        #  confidence: 0.5140249729156494,
        #  bbox_xywh: [1678.99377441  526.4251709    36.55554199   80.11364746],
        #  confidence: 0.5057685971260071]

        # update tracker
        self.tracker.predict()
        # 现在输入的detections 是 做了嵌套编号的 list[ [0编号],[1编号] ]
        self.tracker.update(detections)
        # print("confidence {}".format(detections[0].confidence))

        # output bbox identities
        # tracks 存储相关信息
        outputs = []
        # tracker的属性 trackers储存着 很多个track类实例
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()  # (top left x, top left y, width, height) 每帧都刷新
            x1, y1, x2, y2 = self._xywh_to_xyxy_centernet(
                box)  # xywh 转成 矩形的对角点坐标

            # 画运动轨迹
            # 轨迹为检测框中心
            center = (int((x1+x2)/2),int((y1+y2)/2))#画轨迹图　记录每一次的中心点
            # 轨迹为检测框底部
            # center = (int((x1+x2)/2), int((y2)))  # 画轨迹图　记录每一次的底部

            points[track.track_id].append(center)  # 用队列先进先出的结构　记录运动中心轨迹
            # print(points[1][-1])  # 查看跟踪号为1的对象的中心点存储记忆
            # for j in range(1, len(points[track.track_id])):
            #     if points[track.track_id][j - 1] is None or points[track.track_id][j] is None:
            #        continue
            #     # thickness = int(np.sqrt(32 / float(j + 1)) * 2) #第一个点重　后续线逐渐变细
            #     cv2.line(ori_img,(points[track.track_id][j-1]), (points[track.track_id][j]),(8,196,255),thickness = 3,lineType=cv2.LINE_AA)

            track_id = track.track_id
            confidences = track.confidence * 100
            cls_num = track.class_num
            # print("track_id {} confidences {}".format(track_id,confidences))

            outputs.append(
                np.array([x1, y1, x2, y2, track_id, confidences, cls_num], dtype=np.int))

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs, points

    # for centernet (x1,x2 w,h -> x1,y1,x2,y2)

    def _xywh_to_xyxy_centernet(self, bbox_xywh):
        x1, y1, w, h = bbox_xywh
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(int(x1+w), self.width-1)
        y2 = min(int(y1+h), self.height-1)
        return int(x1), int(y1), x2, y2

    # for yolo  (centerx,centerx, w,h -> x1,y1,x2,y2)
    def _xywh_to_xyxy_yolo(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        # TODO
        features = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy_centernet(box)
            im = ori_img[y1:y2, x1:x2]
            feature = self.extractor(im)[0]
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features


if __name__ == '__main__':
    pass
