# Ideas

> This is the summary of new ideas in CVPR 2018.  This summary is based on author. If you have different idea, ignore this summary and hold on your own idea.



## Graph Generation Network

> A deep learning network generates a graph which contains the node and edges.

- Dynamic Graph Generation Network: Generating Relational Knowledge from Diagrams
  - This is a work for object detection and matching.
  - Author create a dynamic network for solving matching problems
- Some Related Work:
  - [GraphRNN: A Deep Generative Model for Graphs](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) ([Code](https://github.com/JiaxuanYou/graph-generation) [Paper](https://arxiv.org/pdf/1802.08773))



## Point Cloud Auto Encoder

> As we know, point cloud is hard to express. 

- FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation [[Code]()]

> ![1546482453720](./img/1546482453720.png)

![1546482622132](./img/1546482622132.png)

## Point Cloud Segmentation & Recognition

- Attentional ShapeContextNet for Point Cloud Recognition
  - ![1546499335413](img/1546499335413.png)
  - ![1546499369703](img/1546499369703.png)
  - ![1546499461768](img/1546499461768.png)

## Weak Supervision

- Bootstrapping the Performance ofWebly Supervised Semantic Segmentation [[code](https://github.com/ascust/BDWSS)]

  ![1546483318589](img/1546483318589.png)

  ![1546483372251](img/1546483372251.png)

## Image Segmentation

- DenseASPP for Semantic Segmentation in Street Scenes
  - ![1546499912432](img/1546499912432.png)
  - ![1546499988980](img/1546499988980.png)

## RGB Object Detection

- DeepVoting: A Robust and Explainable Deep Network for Semantic Part Detection under Partial Occlusion [[VehicleSemanticPart dataset](https://drive.google.com/file/d/1FU6Jw27yUj5XIVRt1Gj9z6Fb064wI3UE/view?usp=sharing)]
  - ![1546503772412](img/1546503772412.png)
  - ![1546504047693](img/1546504047693.png)
  -  ![1546504577083](img/1546504577083.png)
  - 

## RGB-D Object Detection

> RGB-D camera can output two kind of images: RGB Image and Depth Image. Combining these images can leads to better object detection or segmentation.

- Progressively Complementarity-aware Fusion Network for RGB-D Salient Object Detection
  - ![1546484018647](img/1546484018647.png)
  - ![1546484061551](img/1546484061551.png)

## Face Recognition

- Dynamic Feature Learning for Partial Face Recognition
  - ![1546500502426](img/1546500502426.png)
  - ![1546500529351](img/1546500529351.png)
  - 

## Scene Text Detection & Recognition

- Geometry-Aware Scene Text Detection with Instance Transformation Network
  - ![1546499001962](img/1546499001962.png)
  - ![1546499080758](img/1546499080758.png)

## Dense object tracking

> Object tracking is a fundamental and hard problems. Lots of great work are proposed these years. Dense object tracking is more difficult.

- Towards dense object tracking in a 2D honeybee hive 
  - ![1546484491119](img/1546484491119.png)
  - ![1546484512929](img/1546484512929.png)
  - ![1546484752284](img/1546484752284.png)



## 3D Point Cloud Segmentation

- SPLATNet: Sparse Lattice Networks for Point Cloud Processing [[code](https://github.com/NVlabs/splatnet)]
  - ![1546485228619](img/1546485228619.png)
  - ![1546485248400](img/1546485248400.png)

## 3D Point Cloud Classification

- A Network Architecture for Point Cloud Classification via Automatic Depth Images Generation

  - > A network for 3D Point Cloud Projection whose result is used for classification.
    >
    > My ideas:
    >
    > - Automatic multiple projection (Find the 4 best projection plane which can recover 3D Point Cloud accurately)
    > - Use the proposed projection plane for 3D Point Cloud Segmentation

  - ![1546505131211](img/1546505131211.png)

  - 

## 3D Point Cloud Reflection Removing

- Reflection Removal for Large-Scale 3D Point Cloud
  - ![1546495473154](img/1546495473154.png)
  - ![1546495574950](img/1546495574950.png)
  - 

## Human Pose Estimation & Tracking

- PoseTrack: A Benchmark for Human Pose Estimation and Tracking [[dataset](https://posetrack.net/)]
  - ![1546485484337](img/1546485484337.png)
  - ![1546485523933](img/1546485523933.png)

## Action Recognition

- Recognizing Human Actions as the Evolution of Pose Estimation Maps

  - ![1546496118260](img/1546496118260.png)
  - ![1546496143281](img/1546496143281.png)
  - ![1546496230498](img/1546496230498.png)

- Temporal Deformable Residual Networks for Action Segmentation in Videos

  - > Action Segmentation in Temporal Space

  - ![1546498322582](img/1546498322582.png)

  - ![1546498397386](img/1546498397386.png)

- Unsupervised Learning and Segmentation of Complex Activities from Video

  - ![1546498684216](img/1546498684216.png)
  - ![1546498711697](img/1546498711697.png)

## 3D Map Reconstruction

- InLoc: Indoor Visual Localization with Dense Matching and View Synthesis [[code](http://www.ok.sc.e.titech.ac.jp/INLOC/), [dataset](http://www.ok.sc.e.titech.ac.jp/INLOC/)] 
  - ![1546492635980](img/1546492635980.png)

## 3D Face Reconstruction

- Disentangling Features in 3D Face Shapes for Joint Face Reconstruction and Recognition 

  - ![1546497975846](img/1546497975846.png)
  - ![1546498021526](img/1546498021526.png)

- Alive Caricature from 2D to 3D [[dataset](https://github.com/QianyiWu/ Caricature-Data)]

  - ![1546503168789](img/1546503168789.png)

  - ![1546503328125](img/1546503328125.png)

  - ![1546503353548](img/1546503353548.png)


## Style Transfering

- Learning to Sketch with Shortcut Cycle Consistency 

  - > Translating object image to a sketch

  - ![1546493448354](img/1546493448354.png)

  - ![1546493544758](img/1546493544758.png)

- PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup

  - ![1546496413870](img/1546496413870.png)
  - ![1546496527002](img/1546496527002.png)
  - ![1546496609793](img/1546496609793.png)

## Distort Recover

- Distort-and-Recover: Color Enhancement using Deep Reinforcement Learning [[code](t https://sites.google.com/view/distort-and-recover/)]
  - ![1546496902232](img/1546496902232.png)
  - ![1546496933828](img/1546496933828.png)
- Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation [[code](https:// github.com/yhjo09/VSR-DUF), [dataset](https://media.xiph.org/video/derf/)]
  - ![1546497258838](img/1546497258838.png)
  - ![1546497233950](img/1546497233950.png)
  - ![1546497293210](img/1546497293210.png)
- Missing Slice Recovery for Tensors Using a Low-rank Model in Embedded Space [[code](https://sites.google.com/site/yokotatsuya/ home/software)]
  - ![1546497572543](img/1546497572543.png)

## Video Captioning

- Fine-grained Video Captioning for Sports Narrative 
  - ![1546493968656](img/1546493968656.png)
  - ![1546494002462](img/1546494002462.png)

## Image Captioning

- Neural Baby Talk [[code](https://github.com/jiasenlu/NeuralBabyTalk)]
  - ![1546501396381](img/1546501396381.png)
  - ![1546501506866](img/1546501506866.png)
  - ![1546501548865](img/1546501548865.png)       

## Visual Question Answering

> Visual Question Answering is a sub-field of sequence-to-sequence translation problem which includes 3 objects (or more in the future): images, words, actions.

- Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments [[project](https://bringmeaspoon.org/), [dataset](https://niessner.github.io/Matterport/), [code](https://github.com/peteanderson80/Matterport3DSimulator)]

  - ![Demo](https://bringmeaspoon.org/assets/demo.gif)
  - ![1546494780961](img/1546494780961.png)
  - ![1546494985549](img/1546494985549.png)

- Don’t Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering [[project](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/), [dataset](https://www.cc.gatech.edu/~aagrawal307/vqa-cp/) ]

  - ![1546495291333](img/1546495291333.png)
  - ![1546495326827](img/1546495326827.png)

- Show Me a Story: Towards Coherent neural story illustration

  - > Convert Sentence to Image

  - ![1546500184771](img/1546500184771.png)

  - ![1546500214543](img/1546500214543.png)

  - ![1546500348460](img/1546500348460.png)

## Frame Estimation

- Globally Optimal Inlier Set Maximization for Atlanta Frame Estimation Kyungdon
  - ![1546501243852](img/1546501243852.png)
  - 