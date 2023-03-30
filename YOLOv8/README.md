1) ***PotHole detector*** - Instance segmentation tool, capable of identifying and tagging potholes. Trained on custom dataset of 610 images and validation set of 130 images. Can be used in aerial survey of urban roads if trained for low light application(daylight surveys are tough due to traffic congestion). 
------
About YOLO
-------

YOLO (You Only Look Once) is a popular real-time object detection algorithm that was first introduced by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi in 2016. The algorithm uses a deep neural network to identify and localize objects in an image or a video stream, and it can detect multiple objects in a single frame.
It has proved to be faster and nearly as accurate as moving window approaches, although it uses less space and attains faster fps.

In terms of technicalities, YOLO uses a **deep convolutional neural network (CNN) architecture**, which consists of several layers of convolutional and pooling operations, followed by fully connected layers. The network takes an input image and *produces a set of bounding boxes and class probabilities* for each detected object. The YOLO algorithm is trained using *backpropagation and stochastic gradient descent (SGD)* to minimize the classification and localization errors. However, this also means that it might be less accurate for complex images with small objects and shapes.

YOLO v8 as released by ultralytics introduces a new loss function and has reportedly less training time (~30 % reduction). I find this actually faster than the previous versions (Yolov3, Yolov5) however, I haven't tested this on edge devices like Rapberry Pi. 
