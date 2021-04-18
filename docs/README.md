---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: human-behavior-prediction-using-cctv
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Adaptive people movement and action prediction using CCTV to control appliances

#### Team

- E/15/010, Ruchika Alwis, [email](mailto:e15010@eng.pdn.ac.lk)
- E/15/265, Risith Perera, [email](mailto:risithperera@eng.pdn.ac.lk)
- E/15/347, Isuru Sudasinghe, [email](mailto:isuru.sudasinghe@eng.pdn.ac.lk)

#### Supervisors

- Eng. (Dr.) Kamalanath Samarakoon, [email](mailto:kamalanath@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

With the availability of high-performance processors
and GPUs, the demand for Machine learning, Deep learning
algorithms is growing exponentially. It has become more and
more possible to explore the depths of fields like Computer
vision with these trends. Detecting humans in video footage using
computer vision is one such area. Although human detection is
somewhat primitive when compared to today’s technology, using
that data to produce various results like recognizing postures,
predicting behaviors, predicting paths are very advanced fields
and they have very much room left to grow. Various algorithms,
approaches are available today to accomplish the above kind of
tasks, from classical machine learning, neural networks to statistical
approaches like Bayes theorem, Hidden Markov Models,
Time series, etc. This paper summarize the result of a system
that combines above technologies in order to control electric
appliances through predictions. These predictions are deducted
by analyzing CCTV footages of the user using computer vision.

## Related works

Over the years it can be seen that researchers have tried
various approaches to analyze video footage in order to
produce results like human tracking, path prediction, action
recognition, action/behavior prediction, etc as well as used
existing data to make behavior predictions as well.
There are many inbuilt libraries that are widely used in
these researches such as Alphapose, Openpose [2], VNect
[16] that produce very good results. It can be noticed that
most methods used to detect involves some sort of machine
learning or deep learning algorithms. These models, therefore,
have to be trained with a reasonable amount of data in order
to get good results. Therefore it can be observed that many
propose using already trained algorithms [2] unless the paper
is about improving or proposing new algorithms itself [24].
Another observable factor is that many approaches use various
other techniques prior to the use of a machine learning or
deep learning algorithm in the end. There are instances where
saliency maps [8] are used to enhance the prediction of
path detection algorithms, use of separate feature extraction
algorithms, etc.
On the other hand, prediction is the most difficult part out
of the two aspects mentioned above. Prediction algorithms are
Fig. 1. System Design Diagram
very sensitive to variation in tiny details and produce considerable
deviations. Even the position of camera placement
has a drastic impact on the final result in prediction scenarios
[2]. For prediction approaches it can observe that researchers
have successfully attempted techniques like Convolutional
Neural Networks (CNN), Recurrent Neural Networks(RNN)
all the way to algorithms like Markov Chains, Hidden Markov
Models (HMM), Hierarchical Representation [13], Time Series
Analysis.

## Methodology

As  mentioned  in  the  introduction,  the  complete  systemconsists  of  three  separate  components  that  can  be  run  onthree separate PCs if required. These components are HumanIdentification,  Behavior  Extraction,  and  Behavior  Prediction.Within these subsystems, there are four software componentsas  face  recognition,  human  detection  and  tracking,  actionrecognition, and behavior prediction.A.  Face recognitionDifferent methods of image processing and machine learn-ing  techniques  have  been  used  in  this  component.  Moreweight was taken on improving the accuracy and the executiontime.  Initially,  the  python  face-recognition  package  was  usedwhich has been implemented on top of dlib, which is a C++toolkit containing machine learning algorithms. But it has lowaccuracy and less performance. In order to obtain the desiredoutcome,  the  FaceNet-PyTorch  python  package  is  used  toperform face detection using MTCNN and face identificationusing  an  Inception  Resnet  model.  It  is  pre-trained  on  theVGGFace2 dataset. More advanced implementation details canbe found on the Github Repository - FaceNet-PyTorch.Mainly the whole process is separated into three parts whichare face detection, getting face embeddings, and finally, iden-tifying  faces  by  measuring  the  distance  between  embeddingsin  a  specific  frame.  Face  identification  can  be  easily  appliedto  raw  images  by  first  detecting  faces  using  MTCNN  beforecalculating  embedding  or  probabilities  using  an  InceptionResnet model. It gives huge performance improvement ratherthan sending the entire frame to the Resnet model for gettingface embeddings.B.  Human detection and trackingHere  YOLO  v3  is  used  to  detect  humans.  YOLO  v3  isselected for the detection due to its speed while giving fairlyaccurate  results  at  the  same  time.  Although  it  is  possibleto  detect  humans  using  and  track  them  using  a  model  likeDeepSORT, that approach was discarded due to the overheadcreated when running such heavy models side by side.Therefore the current implementation is, first detect and cropeach person in the frame using the YOLO object classificationmodel.  And  then  send  each  cropped  person  image  to  theReID  model.  Since  this  ReID  model  was  implemented  usingPyTorch, proper PyTorch implementation of YOLO has to beused  instead  of  TensorFlow  implementation.  Because  whendifferent  implementations  of  Tensorflow  and  PyTorch  run  atthe  same  time,  the  GPU  memory  can  be  crashed.  Furtherdetails of that PyTorch YOLO implementation can be found onthe Github Repository - PyTorch-YOLOv3. The PyTorch Reidmodel is based on the works by Zheng et al. There is a Part-based  Convolutional  Baseline  (PCB)  that  conducts  uniformpartition  on  the  conv-layer  for  learning  part-level  features.  Itdoes  not  explicitly  partition  the  images.  PCB  takes  a  wholeimage as the input and outputs a convolutional feature. Beinga  classification  net,  the  architecture  of  PCB  is  concise,  withslight  modifications  on  the  backbone  network.  The  trainingprocedure  is  standard  and  requires  no  bells  and  whistles.  Itshows  that  the  convolutional  descriptor  has  a  much  higherdiscriminative ability than the commonly used fully-connected(FC) descriptor.In order to identify the same person from different perspec-tives, it is required to have a collection of all persons’ imagesthat we are going to identify in our domain. That’s where theface  identification  section  comes  into  place.  Once  a  personhas identified from his face through the entrance camera, that

person’s full-body image is saved in the human database underthe current date folder. This is done because the same personcan wear different clothes on different days. Then periodicallygenerates features of every person in the current day folder andsaves it as a single matrix file.Now  in  the  human  tracking  part,  first,  the  feature  matrixfile  is  loaded  from  the  folder  corresponding  to  the  currentday.  Then  humans  in  the  current  frame  are  detected  usingYOLO and features are extracted from the ReID model. Thesefeatures are compared with each individual feature within thefile collection and find the person tag. This implementation issomewhat similar to the approach we used in face identifica-tion.C.  Action RecognitionTo identify actions performed by each individual person ofthe frame, first, the pose or the skeleton of the person has tobe estimated. To do that, PyTorch implementation of openposeis  used.  It  produces  18  joint  points  of  the  person.  Furtherimplementation details can be found in the Github Repository- PyTorch-openpose. Once the joints points are obtained then itis sent to an action classifier to determine the action. A windowof  five  frames  is  used  to  extract  features  of  body  velocity,normalized  joint  positions,  and  joint  velocities.  Then  meanfiltering  the  prediction  scores  between  2  frames.  Get  a  labelof  the  person  if  the  score  is  larger  than  a  certain  threshold.Here,  a  pre-trained  action  classifier  provided  by  the  GithubRepository  -  Realtime-Action-Recognition  is  used  to  obtainthe action identification.D.  Behavior predictionThere  are  two  Machine  learning  algorithms  used  in  theprediction  model.  One  Algorithm  is  used  to  get  the  nextlocation  while  the  other  algorithm  is  used  to  determine  thestate  of  the  electric  bulbs.  Due  to  the  constraints  in  the  timeframe  to  create  a  totally  new  database  from  scratch,  a  mockdatabase has to be created for this. The Final data-set needs tobe of the structure in Table I. The public data-set we selected tocreate the mock data set has the structure in Table II. This data-set  consists  of  readings  taken  from  motion  sensors,  pressuresensors that were taken in fixed time intervals. This raw datais  then  used  to  predict  the  movement  of  the  owner  insidethe  house.  After  processing  the  data-set  based  on  multipleassumptions, a mock data-set is created that has the requireddata structure as in Table I.IDTimeLocationActionState of appliance 1TABLE IDATASTRUCTURE OF THEREQUIREDDATASETSensor IDTime StampValueNameTABLE IIDATASTRUCTURE OF THERAWDATASETIn  order  to  predict  the  next  location,  the  above  dataset  isprocessed  gain  before  feeding  into  the  classifier.  First  of  allmultiple  sets  of  equences  of  locations  are  made  using  thefollowing inferences.•A sequence of locations would start after one trigger pointto he start of another trigger point.•A set of sequences will always belong to the same day.•A trigger point will be described as following–An instance where the state of an electric appliancechanges.–An  instance where  the person’s  location  changes tothe Bed.These  sequences are  then  used to  train  the HMM  (HiddenMarkov Model). To predict the state of the electric appliance,a catBoost Classifier is used. And the above sequences, alongwith the result from HMM is used to make the prediction onthe state of the devic

## Experiment Setup and Implementation

## Results and Analysis

## Conclusion

The main result for the lack of accuracy is due to the mock
data set that was created to train the model. The original
dataset was consist of random sensor data that was taken in
fixed time intervals. This raw dataset was then modified to
create sequences of movements that would end up specific
tasks such as sleeping in the bed, turning lights on/off, etc.
However, it seems that the resultant sequences have deviated
significantly from a natural routine of a human. This has
resulted in a low accuracy even when an HMM is used to
predict the next location.
Also, it can be observed that the accuracy of the HMM
keeps increasing when the number of hidden states in the
Multinomial HMM kept increasing. But the complexity of the
transformation matrix increases exponentially when the hidden
states are increased. After 20 hidden states, the time to fit the
model for a system with 30 hidden states takes a duration
recorded in hours, which is a huge drawback to a real-time
system that needs to be trained periodically. Therefore, the
model was selected when it showed the highest accuracy of
53% under 19 hidden states.
The reason for obtaining low accuracy even when an LSTM
is used is because of the nature of the dataset. This resultant
dataset lack fixed time steps between adjacent data. And lots of
such redundant data was removed during data preprocessing.
However, it can be assumed that an LSTM would yield much
better results if the data can be recorded in fixed intervals. It
is cleared that after the data pre-processing, the problem has
deviated from the context of a time series analysis.
The aim of this project was to practically attempt the
automation of control of basic electric appliances through the
data obtained by processing CCTV footage data. Even Though
the primary focus was to develop a model with the capability to
successfully do the predictions, it became clear that the system
to extract the data from video footage plays an important
role as well. Therefore the main focus for this project during
this time-frame was to develop a robust system to extract the
required data from a CCTV camera network in real-time.
Finally, for future works, we would like to summarize what
we have discussed above. This project clearly proves the
possibility to automate household electric appliances just by
observing human behavior. However, in order to implement
a system at the domestic level, all the existing components
have to be further fine-tuned. And in order to achieve that
goal, each component has to be optimized separately. We like
to state that this project can be used as the foundation for
such an attempt so that the components of this system can be
separated into parts, developed, and then combined together
for a better, much more accurate system in the future. And
all the models used here can be trained with custom datasets
as well. Therefore the fastest way to improve the performance
would be to train each component with custom datasets that
are specifically designed for this system.

## Publications
1. [Semester 7 report](./)
2. [Semester 7 slides](./)
3. [Semester 8 report](./)
4. [Semester 8 slides](./)
5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./).


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
