# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=1):
        # 三个默认值的意思是最大iou马氏距离是0.7，
        # 最大生命周期的连续帧数是30（区分轨迹是否固定和删除状态），
        # 命中的帧数（区分轨迹是否暂定的（tentative）和固定状态）
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age  # 超过max_age帧没有检测到就销毁了跟踪器
        self.n_init = n_init  # 至少要n_init次才会创建跟踪器

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def __repr__(self):
        return "self.tracks: {}\nself._next_id: {}\n".format(self.tracks, self._next_id)

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        输入的detections 是 做了嵌套编号的 list[ [0编号],[1编号] ]
        detection_idx = 0,1,2,...就是嵌套编号
        detections[detection_idx].tlwh          # (top left x, top left y, width, height) [1507.0871582   487.81103516   33.9420166    82.56787109]
        detections[detection_idx].confidence    # confidence 0.5209447145462036
        detections[detection_idx].feature       # 特征图
        """
        # print(detections)

        # Run matching cascade.
        # 得到匹配对、未匹配的tracks、未匹配的dectections
        matches, unmatched_tracks, unmatched_detections = self._match(
            detections)

        # Update track set.
        # 对于每个匹配成功的track，用其对应的detection进行更新
        for track_idx, detection_idx in matches:
            # track_idx, detection_idx 前者是跟踪号，后者是进来前的编号
            # 注意这里是tracks 的update 而不是 tracker的update
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        # 对于未匹配的成功的track，将其标记为丢失 TODO 这里好像和源码不太一样
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        #  _initiate_track ()
        # 对于未匹配成功的detection，初始化为新的track
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # self.tracks
        # [
        # self.mean: [ 1.53284799e+03  5.28280968e+02  4.03347351e-01  8.42259419e+01
        # -1.39722709e+00 -9.29483560e-02  4.91479257e-09 -3.11543142e-01]
        # self.track_id: 1,
        # self.mean: [ 1.68779628e+03  5.61941406e+02  4.18704876e-01  8.83916252e+01
        # 8.18810003e-01  7.40294860e-01 -1.73386066e-08  2.21596011e-01]
        # self.track_id: 2
        # ]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []  # 这里 重置了features
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
            """
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)
            # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        # 区分开confirmed tracks和unconfirmed tracks
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # 对confirmd tracks进行级联匹配
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # 对级联匹配中未匹配的tracks和unconfirmed tracks中time_since_update为1的tracks进行IOU匹配
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # 整合所有的匹配对和未匹配的tracks
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        # 上面的self._next_id 等同于 当前目标的track_id
        self._next_id += 1
