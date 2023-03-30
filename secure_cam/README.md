Secume_Cam
----------
What you learn - Basics of  computer vision, Image processing (softening, grayscaling, thresholding) using OpenCV, contour/edge detection.

This directory contains files to build a security tool capable of notifying you over mail whenever a person enters your room. 
This has been implemented as follows -
1) OpenCv is used to analyse  grayscaled frames of real time video and compute `frame array differences`. The difference frame becomes input to the next segment.
2) White patches on `frame difference image` indicate objects. Image processing is used to smoothen the raw input and use filters to for countour detection.
3) The specific contour is captured on the real image and a mail is sent containing the image as an attachment.

***Further possible improvements*** -
1) To use this as a realtime tool we need some mechanism to identify friend/intruder. One of many ways is using a binary classification tool on output images. Considering implementation on mobile/edge devices using Tiny YOLO can also be an option.
2) The algorithms can be improved for faster detection. I'm as of now not really sure how but can identify the scooe for betterment.
