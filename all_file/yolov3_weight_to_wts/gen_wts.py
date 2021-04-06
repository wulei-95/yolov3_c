import sys
import struct
import sys
from models import *
from utils.utils import *
import torch


model = Darknet('/home/wl/405/yolov3_weight_to_wts/cfg/yolov3_fc.cfg', (608, 608))
weights = "/home/wl/405/yolov3_weight_to_wts/weights/yolov3_fc_4100.weights"
device = torch_utils.select_device('0')
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)
model = model.eval()

f = open('/home/wl/405/yolov3_weight_to_wts/weights/yolov3_fc_4100.wts', 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

