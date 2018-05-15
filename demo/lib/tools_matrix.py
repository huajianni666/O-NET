import sys
from operator import itemgetter
import numpy as np
import cv2

'''
Function:
	apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
Input:
	rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
Output:
	rectangles: same as input
'''
def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
	return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
	xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
	w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
	if type == 'iom':
	    o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
	else:
	    o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
	pick.append(I[-1])
	I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

'''
Function:
	Filter face position and calibrate bounding box on 12net's output
Input:
	cls_prob  : cls_prob[1] is face possibility
	roi       : roi offset
	pts       : 5 landmark
	rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
	width     : image's origin width
	height    : image's origin height
	threshold : 0.7 can have 94% recall rate on CelebA-database
Output:
	rectangles: face positions and landmarks
'''
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]
    sc  = np.array([prob[pick]]).T
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    w   = x2-x1
    h   = y2-y1
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    pick = []
    for i in range(len(rectangles)):
	x1 = int(max(0     ,rectangles[i][0]))
	y1 = int(max(0     ,rectangles[i][1]))
	x2 = int(min(width ,rectangles[i][2]))
	y2 = int(min(height,rectangles[i][3]))
	if x2>x1 and y2>y1:
	    pick.append([x1,y1,x2,y2,rectangles[i][4],
			 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.7,'iom')

def cls_face_48net(cls_prob,pts,bboxes):
    idx = np.argmax(cls_prob, axis=1)
    conf = np.max(cls_prob,axis =1)
    rectangles=[]
    for item in range(len(idx)):
        if idx[item]!=1:
           rectangle = bboxes[item,:]
           w=rectangle[2]-rectangle[0]
           h=rectangle[3]-rectangle[1]
           sc = conf[item]  
           pt = pts[item,:]
           cls = idx[item]
           rectangles.append([rectangle[0],rectangle[1],rectangle[2],rectangle[3],sc,rectangle[0]+w*pt[0],rectangle[1]+h*pt[1],rectangle[0]+w*pt[2],rectangle[1]+h*pt[3],rectangle[0]+w*pt[4],rectangle[1]+h*pt[5],rectangle[0]+w*pt[6],rectangle[1]+h*pt[7],rectangle[0]+w*pt[8],rectangle[1]+h*pt[9],cls])
    return rectangles
