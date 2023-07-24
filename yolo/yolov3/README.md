Yolov3 
To understand the architecture of this version we need to understand the challenges faced by previous versions(yolo, yolo9000).
Previous versions depended on convolutional/maxpooling layers only to generate feature maps. While this s a good way of retaining important features
it usually *averages the fine details of any image*. This is evident by low performance of such backbones(Darknet 19) on small object detection. 

So, we needed **some changes to retain details and extract rich feature maps.** One way around this was to introduce skip connections. Another trick can be
to avoid unnecessary averaging i.e. to remove pooling layers. This give rise to Darknet 53 ---> CBL + RESIDUALBlocks + minimal pooling !!
