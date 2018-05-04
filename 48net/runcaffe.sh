#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:/workspace/run/huajianni/O-NET/48net
./../../RefineDet/build/tools/caffe train \
	 --solver=solver.prototxt \
  	 --weights=48net.caffemodel \
         --gpu 0,1  2>&1 | tee jobs/onet_train.log
