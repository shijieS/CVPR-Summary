# CVPR Summary
> In this repository, we'll summary some new published datasets, ideas, codes in CVPR these years.

## File Structure

In order to do summary orderly, we use the following file structure.  There are folders named with year. In each folder, there are 4 markdown files (code.md, dataset.md, idea.md, README.md). 

```
├── 2017  (Years)
│   ├── code.md  (new pubished codes)
│   ├── dataset.md (new published datasets)
│   ├── idea.md  (new ideas)
│   ├── img (image for writing markdown files)
│   ├── pdf (raw papers)
│   └── README.md (this year summary)
├── 2018
│   ├── code.md
│   ├── dataset.md
│   ├── idea.md
│   ├── img
│   ├── pdf
│   └── README.md
└── README.md
```



## CVPR2018

### Dataset

- [Old Datasets](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/dataset.md#old-datasets)
- [New Datasets](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/dataset.md#new-datasets)

### Ideas

- [Challenges](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#challenges)
- [My ideas](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#my-ideas)
- [Dataset](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#dataset)
- [Graph Generation Network](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#graph-generation-network)
- [Point Cloud Auto Encoder](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#point-cloud-auto-encoder)
- [Point Cloud Segmentation & Recognition](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#point-cloud-segmentation---recognition)
- [Weak Supervision](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#weak-supervision)
- [Image Classification](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-classification)
- [Image Segmentation](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-segmentation)
- [Image Clustering](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-clustering)
- [Image Compression](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-compression)
- [Image Caption](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-caption)
- [Image Object Detection](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-object-detection)
- [Image Age Estimation](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#image-age-estimation)
- [RGB-D Object Detection](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#rgb-d-object-detection)
- [Stereo Matching](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#stereo-matching)
- [Face Recognition](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#face-recognition)
- [Face Alignment](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#face-alignment)
- [Scene Text Detection & Recognition](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#scene-text-detection---recognition)
- [Dense object tracking](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#dense-object-tracking)
- [Trajectory Prediction](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#trajectory-prediction)
- [Tracking Multiple Object](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#tracking-multiple-object)
- [3D Point Cloud Segmentation](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-point-cloud-segmentation)
- [3D Point Cloud Classification](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-point-cloud-classification)
- [3D Point Cloud Registration](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-point-cloud-registration)
- [3D Point Cloud Reflection Removing](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-point-cloud-reflection-removing)
- [Human Pose Estimation & Tracking](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#human-pose-estimation---tracking)
- [Single Object Tracking](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#single-object-tracking)
- [Action Recognition](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#action-recognition)
- [3D Map Reconstruction](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-map-reconstruction)
- [3D Face Reconstruction](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-face-reconstruction)
- [3D Object Reconstruction](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#3d-object-reconstruction)
- [Style Transfering](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#style-transfering)
- [Distort Recover](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#distort-recover)
- [Video Generation](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#video-generation)
- [Video Captioning](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#video-captioning)
- [Saliency](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#saliency)
- [Visual Question Answering](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#visual-question-answering)
- [Frame Estimation](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#frame-estimation)
- [360 Videos Saliency Prediction](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#360-videos-saliency-prediction)
- [Dynamic network](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#dynamic-network)
- [Ground-to-Aerial Geo-Localization](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#ground-to-aerial-geo-localization)
- [Joint First and Third-Person Videos](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#joint-first-and-third-person-videos)
- [Reasoning Object Affordance](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#reasoning-object-affordance)
- [Fundamental Structured Network](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#fundamental-structured-network)
- [Sound Source Localization](https://github.com/shijieS/CVPR-Summary/blob/ssj-cvpr2018/2018/idea.md#sound-source-localization)

 

## Tasks

| Name      | Detail     | Date     | State |
| --------- | ---------- | -------- | ----- |
| CVPR 2018 | code.md    | 2018.01~ | Doing |
| CVPR 2018 | idea.md    | 2018.01~ | Doing |
| CVPR 2018 | dataset.md | 2018.01~ | Doing |

