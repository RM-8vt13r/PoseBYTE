import numpy as np
import os

from .kalman_filter_pose import KalmanFilter as KalmanFilterPose
from .kalman_filter import KalmanFilter as KalmanFilterBbox
from . import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    def __init__(self, kp_positions, kp_confidences):
        # wait activate
        self._kp_positions = np.array(kp_positions, dtype=float)
        self._kp_confidences = np.array(kp_confidences, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
    
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
    
    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self._kp_confidences = new_track.kp_confidences
    
    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self._kp_confidences = new_track.kp_confidences
    
    @property
    def kp_confidences(self):
        return self._kp_confidences.copy()
    
    @property
    def kp_coco(self):
        return np.concatenate([self.kp_positions.reshape((-1,2)), self.kp_confidences[:,None]], axis=1).flatten()
    
    @property
    def pose(self):
        return {'keypoints': self.kp_coco, 'id': self.track_id}
    
    @property
    def pose_score(self):
        return np.average(self.kp_confidences, weights=self.kp_confidences>0) if np.sum(self.kp_confidences)>0 else 0
    
    @property
    def score(self):
        return self.pose_score
        
    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
        
    @property
    def xyah(self):
        ret = self.tlwh
        ret[:2] += ret[2:]/2
        ret[2] /= ret[3]
        return ret
        
    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
    
class STrackPose(STrack):
    shared_kalman = KalmanFilterPose(0, 0, 0)
    
    def predict(self):
        self.mean, self.covariance = self.kalman_filter.predict(self.mean.copy(), self.covariance.copy())
    
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.array([st.mean.copy() for st in stracks])
            multi_covariance = np.array([st.covariance.copy() for st in stracks])
            STrackPose.shared_kalman.initiate(stracks[0].kp_positions, stracks[0].kp_confidences)
            multi_mean, multi_covariance = STrackPose.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.mean, self.covariance = self.kalman_filter.initiate(self.kp_positions, self.kp_confidences)
    
    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean.copy(), self.covariance.copy(), new_track.kp_positions, new_track.kp_confidences
        )

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean.copy(), self.covariance.copy(), new_track.kp_positions, new_track.kp_confidences
        )
        
    @property
    def kp_positions(self):
        if self.mean is None:
            return self._kp_positions.copy()
        return self.mean[:len(self._kp_positions)].copy() # Mean contains keypoint positions
    
    @property
    def tlwh(self):
        pose_tl = np.array([
            np.min(self.kp_positions[::2][self.kp_confidences>0]),
            np.min(self.kp_positions[1::2][self.kp_confidences>0])
        ])
        pose_wh = np.array([
            np.max(self.kp_positions[::2][self.kp_confidences>0]),
            np.max(self.kp_positions[1::2][self.kp_confidences>0])
        ])-pose_tl
        return np.hstack([pose_tl, pose_wh])
        
class STrackBbox(STrack):
    shared_kalman = KalmanFilterBbox()

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance.copy())
        
    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrackBbox.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        super().activate(kalman_filter, frame_id)
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)
    
    def re_activate(self, new_track, frame_id, new_id):
        super().re_activate(new_track, frame_id, new_id)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean.copy(), self.covariance.copy(), new_track.xyah
        )
        self._kp_positions = new_track.kp_positions

    def update(self, new_track, frame_id):
        super().update(new_track, frame_id)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self._kp_positions = new_track.kp_positions
    
    @property
    def kp_positions(self):
        if self.mean is None:
            return self._kp_positions.copy()
        pose_tl = np.array([
            np.min(self._kp_positions[::2][self._kp_confidences>0]),
            np.min(self._kp_positions[1::2][self._kp_confidences>0])
        ])
        pose_wh = np.array([
            np.max(self._kp_positions[::2][self._kp_confidences>0]),
            np.max(self._kp_positions[1::2][self._kp_confidences>0])
        ])-pose_tl
        return ((self._kp_positions.reshape((-1,2))-pose_tl[None,:])/pose_wh[None,:]*self.tlwh[2:][None,:]+self.tlwh[:2][None,:]).flatten()
    
    @property
    def tlwh(self):
        if self.mean is None:
            pose_tl = np.array([
                np.min(self.kp_positions[::2][self.kp_confidences>0]),
                np.min(self.kp_positions[1::2][self.kp_confidences>0])
            ])
            pose_wh = np.array([
                np.max(self.kp_positions[::2][self.kp_confidences>0]),
                np.max(self.kp_positions[1::2][self.kp_confidences>0])
            ])-pose_tl
            return np.hstack([pose_tl, pose_wh])
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:]/2
        return ret
        

class BYTETracker(object):
    def __init__(self, args, keypoint_falloff, frame_rate=30):
        BaseTrack._count = 0
        
        if args.track_poses:
            STrackPose.shared_kalman = KalmanFilterPose(args.keypoint_confidence_threshold, args.acceleration_memory_factor, args.jerk_memory_factor)
            self.kalman_filter = KalmanFilterPose(args.keypoint_confidence_threshold, args.acceleration_memory_factor, args.jerk_memory_factor)
            self.STrack = STrackPose
            self.distance = matching.oks_distance
        else:
            STrackBbox.shared_kalman = KalmanFilterBbox()
            self.kalman_filter = KalmanFilterBbox()
            self.STrack = STrackBbox
            self.distance = matching.iou_distance

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.args = args
        self.det_thresh = args.track_thresh + 0.1
        self.frame_id = None
        self.buffer_size = int(args.track_buffer_sec * frame_rate)
        self.max_time_lost = self.buffer_size
        self._falloff = keypoint_falloff

    def update(self, output_results, frame):#, img_info, img_size):
        # Output_results: list of detected poses with keypoints
        passed_frames = frame-self.frame_id if self.frame_id is not None else 1
        self.frame_id = frame
        
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        keypoints = np.array([pose['keypoints'] for pose in output_results]).reshape((len(output_results),-1,3)) if len(output_results) else np.zeros(shape=(0,0,3)) # [P,K,3] with P=pose, K=keypoint, 3=x,y,confidence
        keypoint_confidences = keypoints[:,:,2] # [P,K]
        keypoint_coordinates = keypoints[:,:,:2] # [P,K,2]
        scores = np.array([np.average(pose_confidences, weights=pose_confidences>0) if sum(pose_confidences)>0 else 0 for pose_confidences in keypoint_confidences]) # [P]
        
        if len(keypoints):
            remain_inds = scores > self.args.track_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.args.track_thresh

            inds_second = np.logical_and(inds_low, inds_high)
            keypoint_coordinates_second = keypoint_coordinates[inds_second].reshape((sum(inds_second), np.prod(keypoint_coordinates[inds_second].shape[1:]))) # [P,2K]
            keypoint_confidences_second = keypoint_confidences[inds_second] # [P,K]
            keypoint_coordinates = keypoint_coordinates[remain_inds].reshape((sum(remain_inds), np.prod(keypoint_coordinates[remain_inds].shape[1:]))) # [P,2K]
            keypoint_confidences = keypoint_confidences[remain_inds] # [P,K]
            
            scores_second = scores[inds_second]
            scores = scores[remain_inds]
        
        else:
            keypoint_coordinates_second = []
            keypoint_confidences_second = []
            keypoint_coordinates = []
            keypoint_confidences = []
            scores_second = []
            scores = []

        if len(keypoint_coordinates):
            '''Detections'''
            detections = [self.STrack(kp, kps) for kp,kps in zip(keypoint_coordinates, keypoint_confidences)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        for _ in range(passed_frames): self.STrack.multi_predict(strack_pool)
        dists = self.distance(detections, strack_pool, self._falloff).T # Use strack_pool as annotation in OKS calculation
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh_high)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], frame)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(keypoint_coordinates_second):
            '''Detections'''
            detections_second = [self.STrack(kp, kps) for kp,kps in zip(keypoint_coordinates_second, keypoint_confidences_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = self.distance(detections_second, r_tracked_stracks, self._falloff).T
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.args.match_thresh_low)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, frame)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = self.distance(detections, unconfirmed, self._falloff).T
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh_new)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], frame)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, frame)
            activated_stracks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if frame - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, self.distance, self._falloff)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, distance, falloff):
    pdist = distance(stracksb, stracksa, falloff).T # [A,B]
    pairs = np.where(pdist < 0.15) # [A,B]
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
