import sys
sys.path.append('/workspace/run/huajianni/RefineDet/python')
import cv2
import caffe
import numpy as np
import random
import cPickle as pickle
imdb_exit = False

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("#"*rate_num, " "*(100-rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()
################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
    def setup(self, bottom, top):
        self.batch_size = 256
	net_side  = 48

	pos_list  = 'data48/pos_list.txt'
	neg_list  = 'data48/neg_list.txt'
	blur_list = 'data48/blur_list.txt'
	pose_list = 'data48/pose_list.txt'
        cover_list ='data48/cover_list.txt'
	pts_list  = 'data48/pts_list.txt'

	pos_root  = 'data48/pos/'
	neg_root  = 'data48/neg/'
	blur_root = 'data48/blur/'
	pose_root = 'data48/pose/'
	cover_list = 'data48/cover/'
	pts_root = 'data48/pts/'
        self.batch_loader = BatchLoader(pos_list,neg_list,blur_list,pose_list,cover_list,pts_list,net_side,pos_root,neg_root,blur_root,pose_root,cover_list,pts_root)
        top[0].reshape(self.batch_size, 3, net_side, net_side)
        top[1].reshape(self.batch_size, 1)
	top[2].reshape(self.batch_size, 10)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
	loss_task = random.randint(0,5)
        for itt in range(self.batch_size):
            im, label, pts= self.batch_loader.load_next_image(loss_task)
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
	    top[2].data[itt, ...] = pts
    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    def __init__(self,pos_list,neg_list,blur_list,pose_list,cover_list,pts_list,net_side,pos_root,neg_root,blur_root,pose_root,cover_list,pts_root):
	self.mean = 128
        self.im_shape = net_side
        self.pos_root = pos_root
	self.neg_root = neg_root
	self.blur_root = blur_root
	self.pose_root = pose_root
	self.cover_root = cover_root
	self.pts_root = pts_root
	
	self.pos_list = []
	self.neg_list = []
	self.blur_list= []
	self.pose_list= []
	self.cover_list=[]
	self.pts_list = []
	print "Start Reading Pos Data into Memory..."
	if imdb_exit:
	    fid = open('data48/pos.imdb','r')
	    self.pos_list = pickle.load(fid)
	    fid.close()
	else:
	    fid = open(pos_list,'r')
            lines = fid.readlines()
	    fid.close()
	    cur_=0
	    sum_=len(lines)
	    for line in lines:
	        view_bar(cur_, sum_)
	        cur_+=1
	        words = line.split()
	        image_file_name = self.pos_root + words[0] + '.jpg'
	        im = cv2.imread(image_file_name)
	        h,w,ch = im.shape
	        if h!=self.im_shape or w!=self.im_shape:
	            im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
	        im = np.swapaxes(im, 0, 2)
	        im -= self.mean
		label    = int(words[1])
		pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
	        self.pos_list.append([im,label,pts])
	random.shuffle(self.pos_list)
        self.pos_cur = 0
	print "\n",str(len(self.pos_list))," Pos Data have been read into Memory..."


        print "Start Reading Neg Data into Memory..."
        if imdb_exit:
            fid = open('data48/neg.imdb','r')
            self.neg_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(neg_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.neg_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.neg_list.append([im,label,pts])
        random.shuffle(self.neg_list)
        self.neg_cur = 0
        print "\n",str(len(self.neg_list))," Neg Data have been read into Memory..."

        print "Start Reading Blur Data into Memory..."
        if imdb_exit:
            fid = open('data48/blur.imdb','r')
            self.blur_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(blur_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.blur_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.blur_list.append([im,label,pts])
        random.shuffle(self.blur_list)
        self.blur_cur = 0
        print "\n",str(len(self.blur_list))," Blur Data have been read into Memory..."

        print "Start Reading Pose Data into Memory..."
        if imdb_exit:
            fid = open('data48/pose.imdb','r')
            self.pose_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(pose_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.pose_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.pose_list.append([im,label,pts])
        random.shuffle(self.pose_list)
        self.pose_cur = 0
        print "\n",str(len(self.pose_list))," Pose Data have been read into Memory..."


        print "Start Reading Cover Data into Memory..."
        if imdb_exit:
            fid = open('data48/cover.imdb','r')
            self.cover_list = pickle.load(fid)
            fid.close()
        else:
            fid = open(cover_list,'r')
            lines = fid.readlines()
            fid.close()
            cur_=0
            sum_=len(lines)
            for line in lines:
                view_bar(cur_, sum_)
                cur_+=1
                words = line.split()
                image_file_name = self.cover_root + words[0] + '.jpg'
                im = cv2.imread(image_file_name)
                h,w,ch = im.shape
                if h!=self.im_shape or w!=self.im_shape:
                    im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
                im = np.swapaxes(im, 0, 2)
                im -= self.mean
                label    = int(words[1])
                pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
                self.cover_list.append([im,label,pts])
        random.shuffle(self.cover_list)
        self.cover_cur = 0
        print "\n",str(len(self.cover_list))," Cover Data have been read into Memory..."



	print "Start Reading pts-regression Data into Memory..."
	if imdb_exit:
	    fid = open('data48/pts.imdb','r')
	    self.pts_list = pickle.load(fid)
	    fid.close()
	else:
	    fid = open(pts_list,'r')
            lines = fid.readlines()
	    fid.close()
	    cur_=0
	    sum_=len(lines)
	    for line in lines:
	        view_bar(cur_, sum_)
	        cur_+=1
	        words = line.split()
	        image_file_name = self.pts_root + words[0] + '.jpg'
	        im = cv2.imread(image_file_name)
	        h,w,ch = im.shape
	        if h!=self.im_shape or w!=self.im_shape:
	            im = cv2.resize(im,(int(self.im_shape),int(self.im_shape)))
	        im = np.swapaxes(im, 0, 2)
	        im -= self.mean
                label    = int(words[1])
		pts	 = [float(words[ 2]),float(words[ 3]),
			    float(words[ 4]),float(words[ 5]),
			    float(words[ 6]),float(words[ 7]),
			    float(words[ 8]),float(words[ 9]),
			    float(words[10]),float(words[11])]
	        self.pts_list.append([im,label,pts])
	random.shuffle(self.pts_list)
	self.pts_cur = 0 
	print "\n",str(len(self.pts_list))," pts-regression Data have been read into Memory..."

    def load_next_image(self,loss_task): 
	if loss_task == 0:
	    if self.pos_cur == len(self.pos_list):
                self.pos_cur = 0
                random.shuffle(self.pos_list)
            cur_data = self.pos_list[self.pos_cur]  # Get the image index
	    im       = cur_data[0]
            label    = cur_data[1]
	    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
	    if random.choice([0,1])==1:
		im = cv2.flip(im,random.choice([-1,0,1]))
            self.pos_cur += 1
            return im, label, pts

	if loss_task == 1:
	    if self.neg_cur == len(self.neg_list):
                self.neg_cur = 0
                random.shuffle(self.neg_list)
	    cur_data = self.neg_list[self.neg_cur]  # Get the image index
	    im	     = cur_data[0]
            label    = cur_data[1]
	    pts	     = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.neg_cur += 1
            return im, label, pts

        if loss_task == 2:
            if self.blur_cur == len(self.blur_list):
                self.blur_cur = 0
                random.shuffle(self.blur_list)
            cur_data = self.blur_list[self.blur_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.blur_cur += 1
            return im, label, pts

        if loss_task == 3:
            if self.pose_cur == len(self.pose_list):
                self.pose_cur = 0
                random.shuffle(self.pose_list)
            cur_data = self.pose_list[self.pose_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.pose_cur += 1
            return im, label, pts

        if loss_task == 4:
            if self.cover_cur == len(self.cover_list):
                self.cover_cur = 0
                random.shuffle(self.cover_list)
            cur_data = self.cover_list[self.cover_cur]  # Get the image index
            im       = cur_data[0]
            label    = cur_data[1]
            pts      = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.cover_cur += 1
            return im, label, pts

	if loss_task == 5:
	    if self.pts_cur == len(self.pts_list):
                self.pts_cur = 0
                random.shuffle(self.pts_list)
	    cur_data = self.pts_list[self.pts_cur]  # Get the image index
	    im	     = cur_data[0]
            label    = -1
	    pts	     = cur_data[2]
            self.pts_cur += 1
            return im, label, pts
################################################################################
######################Regression Loss Layer By Python###########################
################################################################################
class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	if bottom[0].count != bottom[1].count:
	    raise Exception("Input predict and groundTruth should have same dimension")
	pts = bottom[1].data
	self.valid_index = np.where(pts[:,0] != -1)[0]
	self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
	self.diff[...] = 0
	top[0].data[...] = 0
	if self.N != 0:
	    self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
	for i in range(2):
	    if not propagate_down[i] or self.N==0:
		continue
	    if i == 0:
		sign = 1
	    else:
		sign = -1
	    bottom[i].diff[...] = sign * self.diff / bottom[i].num
################################################################################
#############################Classify Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 2,1,1)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

class cls_Layer(caffe.Layer):
    def setup(self,bottom,top):
	if len(bottom) != 2:
	    raise Exception("Need 2 Inputs")
    def reshape(self,bottom,top):
	label = bottom[1].data
	self.valid_index = np.where(label != -1)[0]
	self.count = len(self.valid_index)
	top[0].reshape(len(bottom[1].data), 5)
	top[1].reshape(len(bottom[1].data), 1)
    def forward(self,bottom,top):
	top[0].data[...][...]=0
	top[1].data[...][...]=0
	top[0].data[0:self.count] = bottom[0].data[self.valid_index]
	top[1].data[0:self.count] = bottom[1].data[self.valid_index]
    def backward(self,top,propagate_down,bottom):
	if propagate_down[0] and self.count!=0:
	    bottom[0].diff[...]=0
	    bottom[0].diff[self.valid_index]=top[0].diff[...]
	if propagate_down[1] and self.count!=0:
	    bottom[1].diff[...]=0
	    bottom[1].diff[self.valid_index]=top[1].diff[...]

