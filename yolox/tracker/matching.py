import cv2
import numpy as np
import scipy
import lap

from cython_bbox import bbox_overlaps as bbox_ious

def okss(estimated_poses, annotated_poses, falloff):
    '''
    Calculate Object Keypoint Similarity values between two sets of poses
    
    Inputs:
    - estimated_poses: estimated poses, size [E] where E is the number of estimated poses
    - annotated_poses: annotated poses, size [A] where A is the number of annotated poses
    - falloff: list of falloff values per keypoint as defined by COCO, size [K] where K is the number of keypoints
    
    Outputs:
    - OKS: matrix with OKS values per pose pair, shape [E,A]
    '''
    OKS_numerator = np.zeros(shape=(len(estimated_poses), len(annotated_poses)), dtype=float)
    OKS_denominator = np.zeros(shape=(len(estimated_poses), len(annotated_poses)), dtype=float)
    for annotation_index, annotated_pose in enumerate(annotated_poses):
        object_scale_squared = (max(annotated_pose['keypoints'][::3])-min(annotated_pose['keypoints'][::3]))*(max(annotated_pose['keypoints'][1::3])-min(annotated_pose['keypoints'][1::3]))
        for estimation_index, estimated_pose in enumerate(estimated_poses):
            joint_distances_squared = (annotated_pose['keypoints'][::3]-estimated_pose['keypoints'][::3])**2 + (annotated_pose['keypoints'][1::3]-estimated_pose['keypoints'][1::3])**2
            annotated_joint_confidences = annotated_pose['keypoints'][2::3]
            OKS_numerator[estimation_index, annotation_index]   = np.sum(np.where((annotated_joint_confidences>0) & (object_scale_squared>0), np.exp(-joint_distances_squared/(2*object_scale_squared*falloff*falloff)), 0))
            OKS_denominator[estimation_index, annotation_index] = np.sum((annotated_joint_confidences>0) & (object_scale_squared>0))
            
    return np.where(OKS_denominator!=0, OKS_numerator/OKS_denominator, 0)

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def oks_distance(atracks, btracks, falloff):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [{'keypoints': track.kp_coco} for track in atracks]
        btlbrs = [{'keypoints': track.kp_coco} for track in btracks]
    _okss = okss(atlbrs, btlbrs, falloff)
    cost_matrix = 1-_okss
    return cost_matrix

def iou_distance(atracks, btracks, args):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
    
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
