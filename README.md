# PoseBYTE
An extension to [BYTE](https://github.com/ifzhang/ByteTrack/tree/main) that uses keypoint detections instead of bounding boxes for multi-object tracking.

# Abstract
Workflow insights can enable safety- and efficiency improvements in the Cardiac Catheterization Laboratory (Cath Lab).
Human pose tracklets from video footage can provide a source of workflow information.
However, occlusions and visual similarity between personnel make the Cath Lab a challenging environment for the re-identification of individuals.
We propose a human pose tracker that addresses these problems specifically, and test it on recordings of real coronary angiograms.
This tracker uses no visual information for re-identification, and instead employs object keypoint similarity between detections and predictions from a third-order motion model.
Algorithm performance is measured on Cath Lab footage using Higher-Order Tracking Accuracy (HOTA).
To evaluate its stability during procedures, this is done separately for five different surgical steps of the procedure.
We achieve up to 0.71 HOTA where tested state-of-the-art pose trackers score up to 0.65 on the used dataset.
We observe that the pose tracker HOTA performance varies with up to 10 percentage point (pp) between workflow phases, where tested state-of-the-art trackers show differences of up to 23 pp.
In addition, the tracker achieves up to 22.5 frames per second, which is 9 frames per second faster than the current state-of-the-art on our setup in the Cath Lab.
The fast and consistent short-term performance of the provided algorithm makes it suitable for use in workflow analysis in the Cath Lab and opens the door to real-time use-cases.

# Usage
The PoseBYTE code is an adaptation of BYTE.
First, clone (https://github.com/ifzhang/ByteTrack/tree/main).
Then, copy the PoseBYTE repository into its root.
This adds the following files into ByteTrack/yolox/tracker:
- byte_tracker_pose.py: a pose-based version of byte_tracker.py.
- kalman_filter_pose.py: a pose-based version of kalman_filter.py.
- matching.py: overrides the original matching.py to include a calculation of Object Keypoint Similarity (OKS).

As an example to use PoseBYTE, we take ByteTrack/tools/demo_track.py as a base.
First, import BYTETracker from yolox.tracker.byte_tracker_pose instead of yolox.tracker.byte_tracker.
Instead of running bounding box detection, run a pose detector like [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) or [OpenPifPaf](https://github.com/openpifpaf/openpifpaf).
Pass a keypoint falloff array to the BYTETracker constructor before the framerate, like those defined by [COCO](https://cocodataset.org/#keypoints-eval).
To the ByteTracker

# Citation
Please cite the following if PoseBYTE aids your work:
```tex
@article{Butler:PoseBYTE,
    author={Butler, Rick M. and Vijfvinkel, Teddy S. and Frassini, Emanuele and van Riel, Sjors and Bachvarov, Chavdar and Constandse, Jan and van der Elst, Maarten and van den Dobbelsteen, John J. and Hendriks, Benno H. W.},
    title={{2D} Human Pose Tracking in the Cardiac Catheterisation Laboratory with {BYTE}},
    journal={},
    year={Submitted},
    month={},
    volume={},
    number={},
    pages={},
    publisher={},
    doi={}
}

@inproceedings{Zhang:Oct22:ByteTrack,
    author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
    title={{ByteTrack}: Multi-object Tracking by Associating Every Detection Box},
    booktitle={Eur. Conf. on Comput. Vis.},
    year={2022},
    month={10},
    pages={1-21},
    publisher={Springer},
    address={Cham, Switzerland},
    doi={https://doi.org/10.1007/978-3-031-20047-2_1}
}
```
