import torch as t
import cv2
from DMRA import RGBNet, DepthNet, ConvLSTM
from RGBT.dataprocessing_predict import testData
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from datetime import datetime
from letters.newxr4 import my64
# from letters.my644 import my644
# from SCRN import SCRN
# from RGBT_net.Twice import Twice
# from RGBT_net.MMNeto3 import MMNet
# from vggbased import VGGBASEDMMNet
# from mma import DEEPSAL,RGBSALBASE,MMA,BOUNDFUSION,SAN_DOUBLEWEIGHT
# from models import HED
test_dataloader = DataLoader(testData, batch_size=1, shuffle=False, num_workers=4)
# for num in np.linspace(0,140,8,dtype=np.uint8):
# os.mkdir('./新的结果-nju2000')
path = '/home/duanting/Desktop/mycode/结果/VT5000/'
isExist = os.path.exists('/home/duanting/Desktop/mycode/结果/VT5000/')
if not isExist:
	os.makedirs('/home/duanting/Desktop/mycode/结果/VT5000/')
else:
	print('path exist')
# 导入模型+
# net = DFNet()
net = my64()
# 导入训练好的参数
# net.load_state_dict(t.load('/media/duanting/shuju/new1/127_mae.pth'))
net.load_state_dict(t.load('/media/duanting/shuju/newxr4/149_mae.pth'))

# params = net.state_dict()
# # keynames = params.keys()
# print(params)
# print(xixi.keys())

# import torch.nn as nn
# from torchvision.models import vgg16_bn
# net = vgg16_bn(pretrained=True)

import torch

with torch.no_grad():
	net.eval()
	net.cuda()
	prec_time = datetime.now()
	for i, sample in enumerate(test_dataloader):

		image = sample['image']
		# imgbound = sample['imgbound']
		# refinedepth = sample['refineDepth']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()
		name = "".join(name)
		# imgbound = Variable(imgbound).cuda()
		# refinedepth = Variable(refinedepth).cuda()
		# hsv = Variable(hsv).cuda()
		# with t.no_grad():
		# pred, loss = net(image)
		# out1 = net(image)
		# out1, out2 = net(image, depth)
		out1 = net(image, depth)

		out1 = F.sigmoid(out1)
		# out2 = F.sigmoid(out2)
		# print(type(pred[5]))
		# out_img = pred[5].data222
		# out = F.softmax(out,dim=1)
		# out2 = F.sigmoid(out2)
		# out3 = F.sigmoid(out3)
		# out4 = F.sigmoid(out4)
		# out5 = F.sigmoid(out5)
		# out6 = F.sigmoid(out6)

		out_img = out1.cpu().detach().numpy()
		# out_img2 = out2.cpu().detach().numpy()

		# out2 = out2.cpu().detach().numpy()
		# out3 = out3.cpu().detach().numpy()
		# out4 = out4.cpu().detach().numpy()
		# out5 = out5.cpu().detach().numpy()
		# out6 = out6.cpu().detach().numpy()


		# out_img_for = out_img[:, 1, :, :]
		# out_back = out_img[:,0,:,:]
		# out_img_l = out_img_for - out_back
		# out_img2 = out2[:, 1, :, :]
		# out_img3 = out3[:, 1, :, :]
		# out_img4 = out4[:, 1, :, :]
		# out_img5 = out5[:, 1, :, :]

		out_img = out_img.squeeze()
		# out_img2 = out_img2.squeeze()
		print(out_img)
		# plt.imshow(out_img, cmap='gray')
		# plt.show()
		plt.imsave(path + name + '.png', arr=out_img, cmap='gray')
		# plt.imsave(path + str(i + 1) + '.png', arr=out_img, cmap='gray')
		# plt.imsave(path + str(i + 1) + 'd.png', arr=out_img2, cmap='gray')
		# plt.imsave(path + str(i + 1) + '_C23DSEP.png', arr=out_img2, cmap='gray')
		# plt.imsave(path + str(i + 1) + '_out2.png', arr=out_img3, cmap='gray')
		# plt.imsave(path + str(i + 1) + '_out3.png', arr=out_img4, cmap='gray')
		# plt.imsave(path + str(i + 1) + '_out4.png', arr=out_img5, cmap='gray')
		# plt.imsave(path + str(i + 1) + '_out5.png', arr=out_img6, cmap='gray')

cur_time = datetime.now()
h, remainder = divmod((cur_time - prec_time).seconds, 3600)
m, s = divmod(remainder, 60)

time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
print(time_str)



