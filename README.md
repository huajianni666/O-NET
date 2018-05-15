# mtcnn-caffe-onet
Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks.

This project provide you a method to update multi-task-loss for multi-input source.
 
![result](https://github.com/huajianni666/O-NET/blob/master/demo/result.jpg)

## Requirement
0. Ubuntu 14.04 or 16.04
1. caffe && pycaffe: [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
2. cPickle && cv2 && numpy 

## Train Data
The training data generate process can refer to [Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)

Sample almost similar to Seanlinx's can be found in `prepare_data`

-  Download Wider Face Training part only from Official Website and unzip to replace `WIDER_train`
-  Cross train among different label databases sharing backbone net.
## Net
The main idea is block backward propagation for different task

48net
![48net](https://github.com/huajianni666/O-NET/blob/master/48net/train48.png)

## Current Status
Huajianni updated in 2018/5/15

