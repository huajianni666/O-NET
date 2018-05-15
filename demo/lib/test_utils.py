import os
import numpy as np
import cv2

def preprocess(src):
    img = cv2.resize(src, (512, 512))
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))
    img[0,:,:] -= 104
    img[1,:,:] -= 117
    img[2,:,:] -= 123
    img = img * 0.017
    return img


def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    #print 'box = ',box
    #print 'cls = ',cls
    #print 'conf= ',conf
    return (box.astype(np.int32), conf, cls)


def preprocess_net48(src):
    src = src.astype(np.float32)
    src = src.transpose((2, 0, 1))
    img = src - 127.5
    img = img * 0.0078125
    return img


def draw(origimg, vis_folder, im_path, rectangles, landmark=False): 
    draw = origimg.copy() 
    font=cv2.FONT_HERSHEY_SIMPLEX
    if landmark :   
        for rectangle in rectangles:
            if rectangle[15]==0:
               cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
               cv2.circle(draw,(int(rectangle[5]),int(rectangle[6])),2,(0,255,0))
               cv2.circle(draw,(int(rectangle[7]),int(rectangle[8])),2,(0,255,0))
               cv2.circle(draw,(int(rectangle[9]),int(rectangle[10])),2,(0,255,0))
               cv2.circle(draw,(int(rectangle[11]),int(rectangle[12])),2,(0,255,0))
               cv2.circle(draw,(int(rectangle[13]),int(rectangle[14])),2,(0,255,0))
            if rectangle[15]==1:
               cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,255,0),2)
            if rectangle[15]==2:
               cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
               cv2.putText(draw, 'blur', (int(rectangle[0]),int(rectangle[1])), font, 1.0, (0,255,0),2)
            if rectangle[15]==3:
               cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
               cv2.putText(draw, 'pose', (int(rectangle[0]),int(rectangle[1])), font, 1.0, (0,255,0),2)
            if rectangle[15]==4:
               cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
               cv2.putText(draw, 'cover', (int(rectangle[0]),int(rectangle[1])), font, 1.0, (0,255,0),2)
        cv2.imwrite(os.path.join(vis_folder, im_path.split('/')[-1]), draw)
    else:
        for rectangle in rectangles:
            cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),2)
        cv2.imwrite(os.path.join(vis_folder, im_path.split('/')[-1]), draw)



