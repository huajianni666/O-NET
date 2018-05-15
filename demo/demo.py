# ------------------------------------------
# AtLab_SSD_Mobilenet_V0.1
# Demo
# by Zhang Xiaoteng
# ------------------------------------------
import numpy as np  
import sys
import os  
from argparse import ArgumentParser
if not '/workspace/run/huajianni/RefineDet/python' in sys.path:
    sys.path.insert(0, '/workspace/run/huajianni/RefineDet/python')
import caffe  
import time
import cv2
from lib.test_utils import preprocess, postprocess, preprocess_net48, draw
from lib.tools_matrix import filter_face_48net, cls_face_48net

def parser():
    parser = ArgumentParser('AtLab SSD Demo!')
    parser.add_argument('--images',dest='im_path',help='Path to the image',
                        default='testimgs',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used',
                        default=0,type=int)
    parser.add_argument('--proto',dest='fdprototxt',help='Fd caffe test prototxt',
                        default='mobile_v2_test_depth_wise_v6_norm_nobn_imgs_deploy.prototxt',type=str)
    parser.add_argument('--model',dest='fdmodel',help='Fd trained caffemodel',
                        default='mobilenet_512x512_v6_norm_bg_iter_31000_nobn.caffemodel',type=str)
    parser.add_argument('--oproto',dest='onetprototxt',help='O-net test prototxt',
                        default='O-Net-deploy.prototxt',type=str)
    parser.add_argument('--omodel',dest='onetmodel',help='O-net caffemodel',
                        default='ONet_iter_300000.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='result',type=str)
    parser.add_argument('--onet',dest='onet',help='Output the keypoints and classes',
                        default=1,type=int)
    return parser.parse_args()



def detect(net, im_path, onet=None, visualize=False, landmark=False, vis_folder=None):

    origimg = cv2.imread(im_path)

    starttime = time.time()

    origin_h,origin_w,ch = origimg.shape
    img = preprocess(origimg)
    img_net48 = origimg.copy()
    
    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(origimg, out)
    endtime = time.time()
    per_time = float(endtime - starttime)
    print '%s speed: {%.3f}s / iter'% (im_path,endtime - starttime)
    cls_dets = np.hstack((box, conf[:, np.newaxis])).astype(np.float32, copy=False)

    if landmark:
        onet.blobs['data'].reshape(len(box),3,48,48)
        crop_number = 0       
	for rectangle in box:
	    #print rectangle
            crop_img = img_net48[max(0,int(rectangle[1])):min(max(0,int(rectangle[3])),origin_h), max(0,int(rectangle[0])):min(max(0,int(rectangle[2])),origin_w)]
            try:
        	    scale_img = cv2.resize(crop_img,(48,48))
        	    #scale_img = np.swapaxes(scale_img, 0, 2)
                    scale_img = scale_img.astype(np.float32)
                    scale_img = scale_img.transpose((2, 0, 1))
                    scale_img = scale_img - 127.5
                    scale_img = scale_img * 0.0078125
        	    onet.blobs['data'].data[crop_number] =scale_img 
        	    crop_number += 1
            except:
	            print 'Couldn\'t find any detections'
        out = onet.forward()
        cls_prob = out['prob']
        pts_prob = out['fc6-pts']   
        #rectangles = filter_face_48net(cls_prob,roi_prob,pts_prob,box,origin_w,origin_h,0.2)
	rectangles = cls_face_48net(cls_prob,pts_prob,box)
        if visualize:
            draw(origimg, vis_folder, im_path, rectangles, landmark=True)           
        return rectangles
    else:
        if visualize:
            draw(origimg, vis_folder, im_path, cls_dets)                       
        return cls_dets, per_time
       
  

if __name__ == "__main__":
    args = parser()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    # caffe.set_mode_cpu()
    assert os.path.isfile(args.fdprototxt),'Please provide a valid path for the fdprototxt!'
    assert os.path.isfile(args.fdmodel),'Please provide a valid path for the fdcaffemodel!'

    net = caffe.Net(args.fdprototxt, args.fdmodel, caffe.TEST)
    net.name = 'AtLab-Refinedet'
    print('Done!')
    
    if args.onet:
        assert os.path.isfile(args.onetprototxt),'Please provide a valid path for the onetprototxt!'
        assert os.path.isfile(args.onetmodel),'Please provide a valid path for the onetcaffemodel!'
        net_48 = caffe.Net(args.onetprototxt,args.onetmodel,caffe.TEST)

    totle_time = 0
    for image in os.listdir(args.im_path):
        img = os.path.join(args.im_path,image)
        if int(args.onet) == 1:                      
            keypoints = detect(net,img,onet=net_48,vis_folder=args.out_path,visualize=True,landmark=True)
        else:
            cls_dets, per_time = detect(net,img,vis_folder=args.out_path,visualize=True,landmark=False)
            totle_time = totle_time + per_time
    print totle_time

