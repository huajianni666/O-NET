import sys
sys.path.append('/workspace/run/huajianni/RefineDet/python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('models/ONet_intial.caffemodel')
solver.solve()
