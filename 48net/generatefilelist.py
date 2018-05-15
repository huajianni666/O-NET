
# -*- coding: utf-8 -*-      
import os  
import argparse

def file_name(file_dir):   
	L=[]   
	for root, dirs, files in os.walk(file_dir):
		for file in files:  
	   		if os.path.splitext(file)[1] == '.jpg':  
	   			L.append(file)  
			if os.path.splitext(file)[1] == '.png':
                                L.append(file)
                        if os.path.splitext(file)[1] == '.jpeg':
                                L.append(file)
	return L 


def parse_args():
  parser = argparse.ArgumentParser(
    description='Generate filelist.txt  ->  ****.jpg 0'
    )
  parser.add_argument(
    '--imgdir', dest='imgdir', default=None, type=str
    )
  parser.add_argument(
    '--filelist', dest='filelist', default=None, type=str
    )
  return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
  print args
  imglist=file_name(args.imgdir)
  with open(args.filelist, "w") as fi:
    for imgname in imglist:
      fi.write(imgname)
      fi.write("\n")
  fi.close()
