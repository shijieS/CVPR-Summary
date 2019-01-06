# Dataset Summary

- [Dataset Summary](#dataset-summary)
  * [Old Datasets](#old-datasets)

  * [New Datasets](#new-datasets)


## Old Datasets

- [THUMOS14](www.baidu.com) 
  - Video Captioning
  - Applied to action recognition and temporal action detection
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
  - presented in CVPR 2012
  - Applied to stereo, optical flow, visual odometry, 3D object detection and 3D tracking
    - Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization
    - WESPE: Weakly Supervised Photo Enhancer for Digital Cameras
- [DPED](http://people.ee.ethz.ch/~ihnatova/)
  - presented in 2017 in the paper 'DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks'
    - WESPE: Weakly Supervised Photo Enhancer for Digital Cameras
- [CityScapes](https://www.cityscapes-dataset.com/)
  - presented in  CVPR 2016 in the paper "The CityScapes Dataset for Semantic Urban Scene Understanding"
  - Urban scene understanding: pixel-level and instance-level semantic labeling.
    - WESPE: Weakly Supervised Photo Enhancer for Digital Cameras
    - Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++
- [Genome](http://visualgenome.org/)
  - presented in 2016 in the paper "Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations"
    - Learning to Segment Every Thing
- [COCO](http://cocodataset.org/)
  - old dataset
    - Learning to Segment Every Thing
- [Panocontext](http://panocontext.cs.princeton.edu/)
  - presented in 2014 in the paper "PanoContext: A Whole-room 3D Context Model for Panoramic Scene Understanding"
    - LayoutNet: Reconstructing the 3D Room Layout from a Single RGB Image
- [CIFAR-10, CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
  - presented in 2009 in the paper "Learning Multiple Layers of Features from Tiny Images"
    - Decorrelated Batch Normalization
    - CondenseNet: An Efficient DenseNet using Learned Group Convolutions
    - Wasserstein Introspective Neural Networks
- [ImageNet](http://www.image-net.org/)
  - old dataset
    - Decorrelated Batch Normalization
    - Analyzing Filters Toward Efficient ConvNet 
- [UCF101](http://crcv.ucf.edu/data/UCF101.php)
  - old dataset
    - What Makes a Video a Video: Analyzing Temporal Information in Video Understanding Models and Datasets

## New Datasets

- [DECADE](https://github.com/ehsanik/dogTorch)
  - a dataset of ego-centric videos from a dog's perspective as well as her corresponding movements.\
  - presented in the paper "Who Let The Dogs Out? Modelling Dog Behaviour From Visual Data "
    - Who Let The Dogs Out? Modelling Dog Behaviour From Visual Data 

- Look
  - a fashion dataset for attribute training
  - presented in 2017 in the paper "Learning the Latent“ Look”: Unsupervised Discovery of a Style-Coherent Embedding from Fashion Images"
    - Creating Capsule Wardrobes from Fashion Images (2018)

- [SuperSloMo](https://github.com/avinashpaliwal/Super-SloMo)
  - There is no specific name for this dataset but it includes 1,132 240 fps video clips, containing 300k video frames.
    - Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

- [Soccer on Your Tabletop](https://github.com/krematas/soccerontable)
  - Image and depth map pairs of soccer players in various body poses and clothing, viewed from a typical soccer game camera. The set includes 12,000 image depth pairs.

- [ModelNet 10-class and 40-class](http://modelnet.cs.princeton.edu/)
  - a comprehensive clean collection of 3D CAD models for objects. 10-class includes 10 categories to train the deep neural network while 40-class includes 40 categories.
  - presented in the paper "RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints"
    - RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints

- [MIRO (Multi View Images of Rotated Objects)](https://github.com/kanezaki/MIRO)
  - includes 10 object instances per object category. It consists of 120 object instances in 12 total categories. 
  - Other datasets such as the RGBD dataset includes an insufficient number of object instances per category and inconsistent cases to the upright orientation assumption.
  - presented in the following paper:
    - RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints

- [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
  - high-quality dataset of youtube video URLs including a diverse range of human focused actions.
  - presented in the paper "The Kinetics Human Action Video Dataset"
    - What Makes a Video a Video: Analyzing Temporal Information in Video Understanding Models and Datasets

- [HPatches](https://github.com/hpatches/hpatches-dataset)
  - patches are extracted from a number of image sequences, where each sequence contains images of the same scenes.
  - presented in CVPR 2017 paper "HPatches: A benchmark and evaluation of handcrafted and learned local descriptors"
    - OATM: Occlusion Aware Template Matching by Consensus Set Maximization

- [CPC (Comparative Photo Composition)](http://www.zijunwei.org/VPN_CVPR2018.html)
  - contains over 1 million comparative view pairs annotated using a cost-effective crowd sourcing work flow.
  - presented in the paper:
    - Good View Hunting: Learning Photo Composition from Dense View Pairs

- [Charades Captions](https://allenai.org/plato/charades/)
  - an extension of the Charades dataset by combining the textual descriptions and sentence scripts verified through AMT (Amazon Mechanical Turk)
  - presented in the paper:
    - Video Captioning via Hierarchical Reinforcement Learning

- [DOTA (Dataset for Object Detection in Aerial Images)](https://captain-whu.github.io/DOTA/)
  - To advance object detection re- search in Earth Vision, also known as Earth Observation and Remote Sensing
  - presented in the paper:
    - DOTA: A Large-scale Dataset for Object Detection in Aerial Images

- Fast and Furious

  - collected by a roof-mounted Li- DAR on top of a vehicle driving around several North- American cities. It consists of 546,658 frames collected from 2762 different scenes. Each scene consists of a continuous sequence.
  - presented in the paper:
    - Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net

- [R2R (Room to Room)](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R)
  - collected 21,567 navigation instructions with
    an average length of 29 words. This is considerably longer than visual question answering datasets where most questions range from four to ten words [4]. However, given the focused nature of the task, the instruction vocabulary is relatively constrained, consisting of around 3.1k words (approximately 1.2k with five or more mentions)
  - presented in the paper:
    - Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments

- [Posetrack](https://posetrack.net/)
  - use the raw videos provided by the popular MPII Human Pose dataset. For each frame in MPII Human Pose dataset we include 41 − 298 neighboring frames from the corresponding raw videos, and then select sequences that rep- resent crowded scenes with multiple articulated people engaging in various dynamic activities.
  - presented in the paper:
    - PoseTrack: A Benchmark for Human Pose Estimation and Tracking
    - Detect-and-Track: Efficient Pose Estimation in Videos

- [Font Dataset](https://github.com/azadis/MC-GAN)
  - collected a dataset including 10K gray-scale Latin fonts each with 26 capital letters.
  - presented in the paper:
    - Multi-Content GAN for Few-Shot Font Style Transfer

- [Sign Language Translation](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/)
  - presents “RWTH-PHOENIX-Weather2014T”, a large vocabulary, continuous SLT corpus. PHOENIX14T is an extension of the PHOENIX14 corpus, which has become the primary benchmark for SLR in recent years. PHOENIX14T constitutes a parallel corpus including sign language videos, sign-gloss annotations and also German translations (spoken by the news anchor), which are all segmented into parallel sentences. 
  - presented in the paper:
    - Neural Sign Language Translation

- FMD (Focus Manipulation Dataset)

  - a focus manipulation dataset (FMD) of images captured with a Canon 60D DSLR and two smartphones having dual lens camera-enabled portrait modes: the iPhone7Plus and the Huawei Mate9. Images from the DSLR represent real shallow DoF images, having been taken with focal lengths in the range 17-70mm and f numbers in the range F/2.8-F/5.6.
  - presented in the paper:
    - Focus Manipulation Detection via Photometric Histogram Analysis

- [See-in-the-Dark Dataset ](https://github.com/cchen156/Learning-to-See-in-the-Dark/tree/master/dataset)
  - a new dataset for training and benchmarking single-image processing of raw low-light images. The See-in-the-Dark (SID) dataset contains 5094 raw short- exposure images, each with a corresponding long-exposure reference image.
  - presented in the paper:
    - Learning to See in the Dark

- [Wild-360](http://aliensunmin.github.io/project/360saliency/http://aliensunmin.github.io/project/360saliency/)
  - challenging 360 videos. One-third of our dataset is annotated with per- frame saliency heatmap for evaluation. Similar to [1, 2], we collect heatmap by aggregating viewers’ trajectories, consisting of 80 viewpoints per-frame.
  - presented in the paper:
    - Cube Padding for Weakly-Supervised Saliency Prediction in 360◦ Videos

- [Functional Map of the World (fMoW)](https://github.com/fMoW/dataset)
  - dataset consists of over 1 million images from over 200 countries1. For each image, we provide at least one bounding box annotation containing one of 63 categories, including a “false detection” category.
  - presented in the paper:
    - Functional Map of the World

- [ADE-Affordance](http://www.cs.utoronto.ca/~cychuang/learning2act/)
  - build our annotations on top of the ADE20K [32]. ADE20k contains images from a wide variety of scene types, ranging from indoor scenes such as airport terminal or living room, to outdoor scenes such as street scene or zoo. It covers altogether 900 scenes, and is a good representative of the diverse world we live in.
  - presented in the paper:
    - Learning to Act Properly: Predicting and Explaining Affordances from Images

- HM-1 and HM-2

  - based on the LSMDC data [35], which contain matched clip-sentence pairs. The LSMDC data contain movie clips and very accurate textual descriptions, which are originally intended for the visually impaired. We generate video and textual sequences in the following way: First, video clips and their descriptions in the same movie are collected sequentially, creating the initial video and text sequences.
  - presented in the paper:
    - A Neural Multi-sequence Alignment TeCHnique (NeuMATCH) 

- [Celeb-ID](https://bdol.github.io/exemplar_gans/)
  - developed an eye in-painting benchmark from high-quality images of celebrities scraped from the web. It contains around 17K individual identities and a total of 100K images, with at least 3 photographs of each celebrity. 
  - presented in the paper:
    - Eye In-Painting with Exemplar Generative Adversarial Networks

- VideoGaze

  - Given a video clip and the annotations of head and eye location, their model combines gaze pathway, saliency pathway and transformation path- way to predict where a person is looking even when the object being looked at is in a different frame. These works only focus on predicting single-person gaze, while do not consider the task of inferring attention shared by multiple persons in social activities.
  - presented in the paper: 
    - Inferring Shared Attention in Social Scene Videos 

- [Daily hand-object actions dataset](https://github.com/hassony2/inria-research-wiki/wiki/first-person-benchmark-dataset)
  - The dataset contains 1,175 action videos belonging to 45 different action categories, in 3 different scenarios, and per- formed by 6 actors. A total of 105,459 RGB-D frames are annotated with accurate hand pose and action category. Action sequences present high inter-subject and intra-subject variability of style, speed, scale, and viewpoint.
  - presented in the paper:
    - First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations

- [’Behance Artistic Media’ (BAM!)](https://bam-dataset.org/)
  - a new in-painting dataset sampled from a dataset of digital artwork.
  - presented in the paper:
    - Disentangling Structure and Aesthetics for Style-aware Image Completion

- [Zalando dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
  - around 19,000 frontal-view woman and top2 clothing image pairs and then removed noisy images with no parsing results, yielding 16,253 pairs. The remaining images are further split into a training set and a testing set with 14,221 and 2,032 pairs respectively.
  - presented in the paper:
    - VITON: An Image-based Virtual Try-on Network

- [wireframe](https://github.com/huangkuns/wireframe)
  - a very large new dataset of over 5,000 images with wireframes thoroughly labelled by humans. The focus is on the structural elements in the image, that is, elements (i.e., line segments) from which meaningful geometric information of the scene can be extracted.
  - presented in the paper:
    - Learning to Parse Wireframes in Images of Man-Made Environments

- Cast In Movies 

  - dataset from 192 movies. We divide each movie into shots using an existing technique.
  - presented in the paper:
    - Unifying Identification and Context Learning for Person Recognition

- [Clipart1k, Watercolor2k, and Comic2k](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets)
  - Each dataset comprises 1,000, 2,000, and 2,000 images of clipart, watercolor, and comic, respectively. The validity of our methods is demonstrated using these datasets.
  - presented in the paper:
    - Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation

- [Uncalibrated Bike Video Dataset](https://github.com/Nick36/unsupervised-learning-depth-ego-motion)
  - a new dataset by recording some videos using a hand-held phone camera while riding a bicycle. This particular camera offers no stabilization. The videos were recorded at 30fps, with a resolution of 720×1280.
  - presented in the paper:
    - Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints

- ACT-X and VQA-X

  - define visual and textual justifications of a classification decision for activity recognition tasks (ACT-X) and for visual question answering tasks (VQA-X).  The MPII Human Pose (MHP) dataset contains 25K images extracted from Youtube videos. From the MHP dataset, we select all images that pertain to 397 activities, resulting in 18, 030 images total.
  - presented in the paper:
    - Multimodal Explanations: Justifying Decisions and Pointing to the Evidence

- [iNaturalist species classification and detection dataset](http://www.cs.cornell.edu/~ycui/publication/cvpr18-inat/)
  - consisting of 859,000 images from over 5,000 different species of plants and animals. It features visually similar species, captured in a wide variety of situations, from all over the world. Images were collected with different camera types, have varying image quality, feature a large class imbalance, and have been verified by multiple citizen scientists.
  - presented in the paper:
    - The iNaturalist Species Classification and Detection Dataset

