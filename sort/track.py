# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    # 具有状态空间的单个目标轨道（x，y，A，H）和相关联速度，
      其中“（x，y）”是bbox的中心，A是高宽比和“H”是高度。

    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean :
            # 初始状态分布的平均向量。
            Mean vector of the initial state distribution.

    covariance :  #协方差
            # 初始状态分布的协方差矩阵
            Covariance matrix of the initial state distribution.

    track_id : int
            # 唯一的轨迹ID
            A unique track identifier.

    n_init : int
            # 在轨道设置为confirmed之前的连续检测帧数。
            当一个miss发生时，轨道状态设置为Deleted帧。
            Number of consecutive detections before the track is confirmed. 
    The track state is set to `Deleted` if a miss occurs within the first
    `n_init` frames.

    max_age : int
            # 在侦测状态设置成Deleted前，最大的连续miss数。
            The maximum number of consecutive misses before the track state is set to `Deleted`.

    feature : Optional[ndarray]
            # 特征向量检测的这条轨道的起源。
            如果为空，则这个特性被添加到'特性'缓存中。
            Feature vector of the detection this track originates from. If not None,
            this feature is added to the `features` cache.

    trackid：
        轨迹ID。
    ---------
    Attributes #属性

    mean : ndarray  #均值：
            # 初始分布均值向量。
            Mean vector of the initial state distribution.

    covariance : ndarray  # 协方差：
        # 初始分布协方差矩阵。
            Covariance matrix of the initial state distribution.

    track_id : int
            A unique track identifier.

    hits : int
            #测量更新的总数。
            Total number of measurement updates.

    hit_streak : int
            # 自上次miss之后，连续测量更新的总数。（更新一次+1）
            Total number of consective measurement updates since last miss.
        age : int
            #从开始的总帧数
            Total number of frames since first occurance.

        time_since_update : int
            # 从上次的测量更新完后，统计的总帧数
            Total number of frames since last measurement update.

    state : TrackState
            # 当前的侦测状态
            The current track state.

    features : List[ndarray]
            # 特性的缓存。在每个度量更新中，相关的特性
    向量添加到这个列表中。
            A cache of features. On each measurement update, the associated feature
            vector is added to this list.

    print ( self.mean , self.track_id )

    [
    self.mean: [ 1.53284799e+03  5.28280968e+02  4.03347351e-01  8.42259419e+01
    -1.39722709e+00 -9.29483560e-02  4.91479257e-09 -3.11543142e-01]
    self.track_id: 1,
    self.mean: [ 1.68779628e+03  5.61941406e+02  4.18704876e-01  8.83916252e+01
    8.18810003e-01  7.40294860e-01 -1.73386066e-08  2.21596011e-01]
    self.track_id: 2
    ]

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.confidence = 0  # 新增
        self.class_num = 0  # 新增
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def __repr__(self):
        return "self.mean: {}\nself.track_id: {}\n".format(self.mean, self.track_id)

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        self.confidence = detection.confidence
        self.class_num = detection.cls_num

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
