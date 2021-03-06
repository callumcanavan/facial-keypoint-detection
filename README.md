# facial-keypoint-detection

A convolutional neural network (CNN) for detecting 68 keypoints on facial images after detection with a pretrained Haar cascade. The network is trained on a 3462 colour images from [Youtube Faces DB](https://www.cs.tau.ac.il/~wolf/ytfaces/). 

Example outputs from an unseen image (after Haar cascade face detection and forward pass through the CNN) are shown below.

![alt text](https://github.com/callumcanavan/facial-keypoint-detection/blob/master/images/cascade.png?raw=true)

![alt text](https://github.com/callumcanavan/facial-keypoint-detection/blob/master/images/keypoints.png?raw=true)

This project was completed as part of the Udacity Computer Vision nanodegree which provided the notebooks and several other functionalities.

# Depends
```
pytorch cv2 numpy
```
