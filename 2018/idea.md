# Ideas

> This is the summary of new ideas in CVPR 2018.  This summary is based on author. If you have different idea, ignore this summary and hold on your own idea.

## Challenges

- [EvalAI](http://evalai.cloudcv.org/)

  - > EvalAI is an open-source web platform for organizing and participating in AI challenges. EvalAI = evaluating the state of the art in AI.

  - ![1546524351806](img/1546524351806.png)

- [VQA Challenge](https://visualqa.org/)

  - > VQA is a new dataset containing open-ended questions about images. These
    > questions require an understanding of vision, language and commonsense 
    > knowledge to answer.

  - ![1546524536960](img/1546524536960.png)


## My ideas

- Video Frame Clustering 
  - Clustering the video frames based on the viewpoint or action or  something else.
  - The dataset can be selected from the action recognition dataset.

## Dataset

- The iNaturalist Species Classification and Detection Dataset

  - ![1546587706837](img/1546587706837.png)
  - ![1546587746447](img/1546587746447.png)

- WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted Pedestrian Detection [[dataset](https://cvlab.epfl.ch/data/data-wildtrack/)]

  - > My Ideas:
    >
    > - We can use this dataset for multiple object tracking.

  - ![1546589369309](img/1546589369309.png)

## Graph Generation Network

> A deep learning network generates a graph which contains the node and edges.

- Dynamic Graph Generation Network: Generating Relational Knowledge from Diagrams
  - This is a work for object detection and matching.
  - Author create a dynamic network for solving matching problems
- Some Related Work:
  - [GraphRNN: A Deep Generative Model for Graphs](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) ([Code](https://github.com/JiaxuanYou/graph-generation) [Paper](https://arxiv.org/pdf/1802.08773))
- Using Compressed Video When do video understanding, such as video caption or action recognition.
- Using RGB-D Camera to have a research on the stereo matching.



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

## Image Classification

- Learning to Understand Image Blur
  - ![1546570478435](img/1546570478435.png)
  - ![1546570493966](img/1546570493966.png)

## Image Segmentation

- DenseASPP for Semantic Segmentation in Street Scenes
  - ![1546499912432](img/1546499912432.png)
  - ![1546499988980](img/1546499988980.png)
- Recurrent Pixel Embedding for Instance Grouping [[project](http://www.ics.uci.edu/~skong2/SMMMSG.html), [code](https://github.com/aimerykong/Recurrent-Pixel-Embedding-for-Instance-Grouping)]
  - ![1546570188744](img/1546570188744.png)
  - ![1546570262076](img/1546570262076.png)
- Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation
  - ![1546571946917](img/1546571946917.png)
  - ![1546571975398](img/1546571975398.png)
- Weakly and Semi Supervised Human Body Part Parsing via Pose-Guided Knowledge Transfer
  - ![1546587504391](img/1546587504391.png)
  - ![1546587521935](img/1546587521935.png)
  - ![1546587576880](img/1546587576880.png)

## Image Clustering

> Clustering is similar with Classification. They have some difference:
>
> - Cluster is the unsupervised learning while classification is supervised learning
> - Clustering is group similar instance based on the features while classification is assigning predefined tags to the instance.
> - ![img](img/Difference-Between-Clustering-and-Classification-in-Tabular-Form.jpg) 

- Dimensionality’s Blessing: Clustering Images by Underlying Distribution [[project](http://www.kind-of-works.com/source_code.html), [code](http://www.kind-of-works.com/code/Clustering_w_Nevlad.zip), [data](http://www.kind-of-works.com/images/release_data.zip)]
  - ![1546563227033](img/1546563227033.png)
  - ![1546563561641](img/1546563561641.png)

## Image Compression

- Learning Convolutional Networks for Content-weighted Image Compression [[project](http://www2.comp.polyu.edu.hk/~15903062r/content-weighted-image-compression.html), [code](https://github.com/limuhit/ImageCompression)]
  - ![1546567936540](img/1546567936540.png)

## Image Caption

- Neural Baby Talk [[code](https://github.com/jiasenlu/NeuralBabyTalk)]
  - ​	![1546501396381](img/1546501396381.png)
  - ![1546501506866](img/1546501506866.png)
  - ![1546501548865](img/1546501548865.png)

- Unsupervised Textual Grounding: Linking Words to Image Concepts
  - ![1546571629853](img/1546571629853.png)
  - ![1546571647694](img/1546571647694.png)

- GroupCap: Group-based Image Captioning with Structured Relevance and Diversity Constraints

  - ![1546587956058](img/1546587956058.png)
  - ![1546587980275](img/1546587980275.png)
  - ![1546588009215](img/1546588009215.png)


## Image Object Detection

- DeepVoting: A Robust and Explainable Deep Network for Semantic Part Detection under Partial Occlusion [[VehicleSemanticPart dataset](https://drive.google.com/file/d/1FU6Jw27yUj5XIVRt1Gj9z6Fb064wI3UE/view?usp=sharing)]
  - ![1546503772412](img/1546503772412.png)
  - ![1546504047693](img/1546504047693.png)
  - ![1546504577083](img/1546504577083.png)

## Image Age Estimation

- Deep Cost-Sensitive and Order-Preserving Feature Learning for Cross-Population Age Estimation
  - ![1546586607984](img/1546586607984.png)
  - ![1546586624218](img/1546586624218.png)
  - 

## RGB-D Object Detection

> RGB-D camera can output two kind of images: RGB Image and Depth Image. Combining these images can leads to better object detection or segmentation.

- Progressively Complementarity-aware Fusion Network for RGB-D Salient Object Detection
  - ![1546484018647](img/1546484018647.png)
  - ![1546484061551](img/1546484061551.png)



- DoubleFusion: Real-time Capture of Human Performances with Inner Body Shapes from a Single Depth Sensor
  - ![1546590633930](img/1546590633930.png)
  - ![1546590712023](img/1546590712023.png)
  - ![1546590799865](img/1546590799865.png)

## Stereo Matching

- **Zoom and Learn: Generalizing Deep Stereo Matching to Novel Domains**
  - ![1546566399766](img/1546566399766.png)
  - ![1546566434232](img/1546566434232.png)

## Face Recognition

- Dynamic Feature Learning for Partial Face Recognition
  - ![1546500502426](img/1546500502426.png)
  - ![1546500529351](img/1546500529351.png)

## Face Alignment

- Disentangling 3D Pose in A Dendritic CNN for Unconstrained 2D Face Alignment
  - ![1546586483068](img/1546586483068.png)
  - ![1546586499053](img/1546586499053.png)
  - ![1546586523080](img/1546586523080.png)

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

## Trajectory Prediction

> I think solving multiple object tracking contains: detection, matching, association and prediction.

- Encoding Crowd Interaction with Deep Neural Network for Pedestrian Trajectory Prediction
  - ![1546517250365](img/1546517250365.png)
  - ![1546517263898](img/1546517263898.png)
  - ![1546517308170](img/1546517308170.png)



## Tracking Multiple Object

- A Causal And-Or Graph Model for Visibility Fluent Reasoning in Tracking Interacting Objects [[project](https://cvlab.epfl.ch/research/research-surv/trackinteractobj/), [code](https://cvlab.epfl.ch/research/research-surv/trackinteractobj/), [dataset](https://cvlab.epfl.ch/research/research-surv/trackinteractobj/)]
  - ![1546587351543](img/1546587351543.png)
  - ![1546587329271](img/1546587329271.png)

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

  - ![1546505430686](img/1546505430686.png)

## 3D Point Cloud Registration

- Inverse Composition Discriminative Optimization for Point Cloud Registration
  - ![1546525137105](img/1546525137105.png)
  - 

## 3D Point Cloud Reflection Removing

- Reflection Removal for Large-Scale 3D Point Cloud
  - ![1546495473154](img/1546495473154.png)
  - ![1546495574950](img/1546495574950.png)

## Human Pose Estimation & Tracking

- PoseTrack: A Benchmark for Human Pose Estimation and Tracking [[dataset](https://posetrack.net/)]
  - ![1546485484337](img/1546485484337.png)
  - ![1546485523933](img/1546485523933.png)

## Single Object Tracking

- Efficient Diverse Ensemble for Discriminative Co-Tracking [[code](http://ishiilab.jp/member/meshgi-k/dedt.html)]

  - ![1546523434608](img/1546523434608.png)
  - ![1546523457537](img/1546523457537.png)
- Correlation Tracking via Joint Discrimination and Reliability Learning
  - ![1546590362472](img/1546590362472.png)
  - ![1546590394492](img/1546590394492.png)
  - 


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

- Learning Latent Super-Events to Detect Multiple Activities in Videos

  - ![1546565540258](img/1546565540258.png)
  - ![1546565607745](img/1546565607745.png)
  - ![1546565649874](img/1546565649874.png)

- Compressed Video Action Recognition

  - > Do Action Recognition on compressed video directly.

  - ![1546571108115](img/1546571108115.png)

- PoTion: Pose MoTion Representation for Action Recognition Vasileios

  - ![1546586789383](img/1546586789383.png)
  - ![1546586827875](img/1546586827875.png)

- One-shot Action Localization by Learning Sequence Matching Network

  - ![1546587828489](img/1546587828489.png)
  - ![1546587842568](img/1546587842568.png)
  - ![1546587868863](img/1546587868863.png)

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
- Extreme 3D Face Reconstruction: Seeing Through Occlusions [[code](github.com/anhttran/extreme_3d_faces)]
  - ![1546524706759](img/1546524706759.png)
  - ![1546524753922](img/1546524753922.png)

## 3D Object Reconstruction

- Im2Struct: Recovering 3D Shape Structure from a Single RGB Image
  - ![1546584443005](img/1546584443005.png)
  - ![1546584486828](img/1546584486828.png)
  - ![1546584828681](img/1546584828681.png)

## Style Transfering

- Learning to Sketch with Shortcut Cycle Consistency 

  - > Translating object image to a sketch

  - ![1546493448354](img/1546493448354.png)

  - ![1546493544758](img/1546493544758.png)

- PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup

  - ![1546496413870](img/1546496413870.png)
  - ![1546496527002](img/1546496527002.png)
  - ![1546496609793](img/1546496609793.png)

- Translating and Segmenting Multimodal Medical Volumes with Cycle- and Shape-Consistency Generative Adversarial Network

  - ![1546588790063](img/1546588790063.png)
  - ![1546588809193](img/1546588809193.png)
  - ![1546588833156](img/1546588833156.png)

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

- Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform [[code](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/), [dataset](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)]

  - ![1546521908899](img/1546521908899.png)
  - ![1546521924643](img/1546521924643.png)

- PhaseNet for Video Frame Interpolation [[video](https://www.youtube.com/watch?v=3zfV0Y7rwoQ)]

  - > Video frame interpolation is a kind of distort recovery which recover the missing information.

  - ![1546525385613](img/1546525385613.png)

  - ![1546525423768](img/1546525423768.png)

- Learning to Extract a Video Sequence from a Single Motion-Blurred Image

  - > A single motion-blurred image contains the information of multiple motion frames. It's very interesting to extract multiple frames from it.

  - ![1546516230184](img/1546516230184.png)

  - ![1546516347214](img/1546516347214.png)

- Crafting a Toolchain for Image Restoration by Deep Reinforcement Learning [[code](https://github.com/yuke93/RL-Restore), ]

  - ![img](img/framework.png)
  - ![img](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/support/restore.gif)

- Learning Dual Convolutional Neural Networks for Low-Level Vision Jinshan

  - ![1546589011455](img/1546589011455.png)
  - ![1546589052425](img/1546589052425.png)

## Video Generation

- **MoCoGAN: Decomposing Motion and Content for Video Generation** [[code](https://github.com/sergeytulyakov/mocogan)]
  - ![TaiChi](https://github.com/sergeytulyakov/mocogan/raw/master/doc/taichi.gif)
  - ![1546569569677](img/1546569569677.png)

## Video Captioning

- Fine-grained Video Captioning for Sports Narrative 

  - ![1546493968656](img/1546493968656.png)
  - ![1546494002462](img/1546494002462.png)

- End-to-End Dense Video Captioning with Masked Transformer

  - > Dense Video means **Untrimmed Video**

  - ![1546518289378](img/1546518289378.png)

  - ![1546518312474](img/1546518312474.png)

- **Bidirectional Attentive Fusion with Context Gating for Dense Video Captioning** [[project](https://cs.stanford.edu/people/ranjaykrishna/densevid/), [dataset](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip)]

  - ![1546568972598](img/1546568972598.png)
  - ![1546568990169](img/1546568990169.png)
  - ![1546569019126](img/1546569019126.png)       

- Neural Sign Language Translation Necati [[code](https://github.com/neccam/nslt), [dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)]
  - ![1546571367693](img/1546571367693.png)
  - ![1546571385676](img/1546571385676.png)

## Saliency

> Video Saliency Detection is very important in pre-processing stage and can be applied widely.
>
> My ideas:
>
> - Saliency detection should combined with questions.
> - Saliency detection seems to be helpful for object detection or segmentation.

- Revisiting Video Saliency: A Large-scale Benchmark and a New Model [[code](https://github.com/wenguanwang/DHF1K), [dataset](https://github.com/wenguanwang/DHF1K)]
  - ![1546518076632](img/1546518076632.png)
  - ![1546518094195](img/1546518094195.png)
- Progressive Attention Guided Recurrent Network for Salient Object Detection
  - ![1546563676585](img/1546563676585.png)
  - ![1546565256829](img/1546565256829.png)
  - ![1546565301958](img/1546565301958.png)

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

- **Multi-Label Zero-Shot Learning with Structured Knowledge Graphs**

  - >  A network for predicting multiple unseen class labels and **knowledge graphs**

  - ![1546515738445](img/1546515738445.png)

  - ![1546515804435](img/1546515804435.png)

  - ![1546515898626](img/1546515898626.png)

  - ![1546515948495](img/1546515948495.png)

- **Transparency by Design: Closing the Gap Between Performance and Interpretability in Visual Reasoning** [[code](https://github.com/davidmascharka/tbd-nets), [dataset](http://cs.stanford.edu/people/jcjohns/clevr/)]

  - ![1546522934380](img/1546522934380.png)
  - ![1546523234505](img/1546523234505.png)

- Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge [[VQT Challenge](https://visualqa.org/)]

  - ![1546523989927](img/1546523989927.png)

  - ![1546524022821](img/1546524022821.png)


## Frame Estimation

- Globally Optimal Inlier Set Maximization for Atlanta Frame Estimation Kyungdon
  - ![1546501243852](img/1546501243852.png)

## 360 Videos Saliency Prediction

- Cube Padding for Weakly-Supervised Saliency Prediction in 360◦ Videos
  - ![1546507243854](img/1546507243854.png)
  - ![1546507263590](img/1546507263590.png)
  - ![1546507303396](img/1546507303396.png)

## Dynamic network

> Dynamic network is the network which outputs different dimension or use different branch inner this network.
>
> My ideas:
>
> - Have a research on the branch selection. We can create a functional branch and do the transfer learning.
> - Make the output dynamic

- HydraNets: Specialized Dynamic Architectures for Efficient Inference

  - ![1546516803332](img/1546516803332.png)

  - ![1546516914934](img/1546516914934.png)


## Ground-to-Aerial Geo-Localization

> Mapping the Ground to the Satellite Map or opposite direction. Or extract more information from this mapping.

- CVM-Net: Cross-View Matching Network for Image-Based Ground-to-Aerial Geo-Localization
  - ![1546518750248](img/1546518750248.png)
  - ![1546521599730](img/1546521599730.png)
  - ![1546521686488](img/1546521686488.png)

## Joint First and Third-Person Videos

> First person videos and third person videos can get matched based on the provided information.
>
> My Ideas:
>
> - There are lots of dataset for this issues, such as Multiple View Object Tracking Dataset
> - This is a new problems it needs more work. 

- **Actor and Observer: Joint Modeling of First and Third-Person Videos** [[code](github.com/gsig/actor-observer)]
  - ![1546566759644](img/1546566759644.png)
  - ![1546566778671](img/1546566778671.png)
  - ![1546566804424](img/1546566804424.png)
  - ![1546566825562](img/1546566825562.png)

## Reasoning Object Affordance

- **Demo2Vec: Reasoning Object Affordances from Online Videos** [[dataset](https://github.com/StanfordVL/opra)]
  - ![1546568212731](img/1546568212731.png)
  - ![1546568233539](img/1546568233539.png)
  - ![1546568301958](img/1546568301958.png)
  - ![1546568331267](img/1546568331267.png)

## Fundamental Structured Network

- **Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks** [[code](https://github.com/TomVeniat/bsn)]
  - ![1546577928464](img/1546577928464.png)
  - ![1546577958258](img/1546577958258.png)
  - ![1546578009218](img/1546578009218.png)
  - ![1546578037589](img/1546578037589.png)

- In-Place Activated BatchNorm for Memory-Optimized Training of DNNs [[code](https://github.com/ mapillary/inplace_abn)]

- Recurrent Residual Module for Fast Inference in Videos

  - ![1546588629165](img/1546588629165.png)

- **NAG: Network for Adversary Generation** [[code](https://github.com/val-iisc/nag)]

  - > This work is an attempt to explore the manifold of perturbations that 
    > can cause CNN based classifiers to behave absurdly. At present, this 
    > repository provides the facility to train the generator that can produce
    > perturbations to fool VGG F, VGG 16, VGG 19, GoogleNet, CaffeNet, 
    > ResNet 50, ResNet 152.

  - ![img](img/nag.png)

- Non-local Neural Networks [[code](https://github.com/titu1994/keras-non-local-nets.git)]

  - ![1546590065781](img/1546590065781.png)
  - ![1546590081842](img/1546590081842.png)

## Sound Source Localization

- Learning to Localize Sound Source in Visual Scenes
  - ![1546588205767](img/1546588205767.png)
  - ![1546588218619](img/1546588218619.png)
  - ![1546588262717](img/1546588262717.png)