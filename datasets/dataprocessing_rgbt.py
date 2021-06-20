from PIL import Image
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import RGBT.data_transform as data_transform
import imageio

# import skimage.io
# import cv2
# Train(一个)
# gt 为png  其他两个为jpg格式

lrimgs = os.listdir('../data3/train/RGB')
lrimgs = [os.path.join('../data3/train/RGB', img) for img in lrimgs]
lrimgs.sort()

gt = os.listdir('../data3/train/GT')
gt = [os.path.join('../data3/train/GT', gtimg) for gtimg in gt]
gt.sort()

depth = os.listdir('../data3/train/T')
depth = [os.path.join('../data3/train/T', dep) for dep in depth]
depth.sort()

bound = os.listdir('../data3/train/bound')
bound = [os.path.join('../data3/train/bound', bod) for bod in bound]
bound.sort()

# val（一个）
# RGBval = os.listdir('../data3/val/RGB')
# RGBval = [os.path.join('../data3/val/RGB', img) for img in RGBval]
# RGBval = sorted(RGBval, key=lambda x: int(x.split('/')[-1].split('.')[0]))
#
# gtval = os.listdir('../data3/val/GTbinary')
# gtval = [os.path.join('../data3/val/GTbinary', gtimg) for gtimg in gtval]
# gtval = sorted(gtval, key=lambda x: int(x.split('/')[-1].split('.')[0]))
#
# depthval = os.listdir('../data3/val/T')
# depthval = [os.path.join('../data3/val/T', dep) for dep in depthval]
# depthval = sorted(depthval, key=lambda x: int(x.split('/')[-1].split('.')[0]))
#
# boundval = os.listdir('../data3/val/GTB')
# boundval = [os.path.join('../data3/val/GTB', bod) for bod in boundval]
# boundval = sorted(boundval, key=lambda x: int(x.split('/')[-1].split('.')[0]))

# test（3个）
# VT800
RGBVT800Test = os.listdir('../data3/test/VT800/RGB')
RGBVT800Test = [os.path.join('../data3/test/VT800/RGB', img) for img in RGBVT800Test]
RGBVT800Test = sorted(RGBVT800Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
gtVT800Test = os.listdir('../data3/test/VT800/GT')
gtVT800Test = [os.path.join('../data3/test/VT800/GT', gtimg) for gtimg in gtVT800Test]
gtVT800Test = sorted(gtVT800Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
depthVT800Test = os.listdir('../data3/test/VT800/T')
depthVT800Test = [os.path.join('../data3/test/VT800/T', dep) for dep in depthVT800Test]
depthVT800Test = sorted(depthVT800Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
# boundTest = os.listdir('./data/test/bound')
# boundTest = [os.path.join('./data/test/bound', bod) for bod in boundTest]
# boundTest = sorted(boundTest, key=lambda x: int(x.split('/')[-1].split('.')[0]))

# VT1000
RGBVT1000Test = os.listdir('../data3/test/VT1000/RGB')
RGBVT1000Test = [os.path.join('../data3/test/VT1000/RGB', img) for img in RGBVT1000Test]
RGBVT1000Test = sorted(RGBVT1000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
gtVT1000Test = os.listdir('../data3/test/VT1000/GT')
gtVT1000Test = [os.path.join('../data3/test/VT1000/GT', gtimg) for gtimg in gtVT1000Test]
gtVT1000Test = sorted(gtVT1000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
depthVT1000Test = os.listdir('../data3/test/VT1000/T')
depthVT1000Test = [os.path.join('../data3/test/VT1000/T', dep) for dep in depthVT1000Test]
depthVT1000Test = sorted(depthVT1000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))

# VT5000
RGBVT5000Test = os.listdir('../data3/test/VT5000/RGB')
RGBVT5000Test = [os.path.join('../data3/test/VT5000/RGB', img) for img in RGBVT5000Test]
RGBVT5000Test = sorted(RGBVT5000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
gtVT5000Test = os.listdir('../data3/test/VT5000/GT')
gtVT5000Test = [os.path.join('../data3/test/VT5000/GT', gtimg) for gtimg in gtVT5000Test]
gtVT5000Test = sorted(gtVT5000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))
depthVT5000Test = os.listdir('../data3/test/VT5000/T')
depthVT5000Test = [os.path.join('../data3/test/VT5000/T', dep) for dep in depthVT5000Test]
depthVT5000Test = sorted(depthVT5000Test, key=lambda x: int(x.split('/')[-1].split('.')[0]))


# error = 0
# for i in range(1588):
# 	lrnum = lr[i][lr[i].split('.')[1].rfind('/')+2:].split('.')[0]
# 	gtnum = gt[i][gt[i].split('.')[1].rfind('/')+2:].split('.')[0]
# 	depthnum = depth[i][depth[i].split('.')[1].rfind('/')+2:].split('.')[0]
# 	# print(lrnum,gtnum,depthnum)
# 	if lrnum != gtnum or gtnum != depthnum:
# 		error += 1
#
# print(error)
#
# #397
# error1 = 0
# for i in range(397):
# 	lrnum = lrTest[i][lrTest[i].split('.')[1].rfind('/') + 2:].split('.')[0]
# 	gtnum = gtTest[i][gtTest[i].split('.')[1].rfind('/') + 2:].split('.')[0]
# 	depthnum = depthTest[i][depthTest[i].split('.')[1].rfind('/') + 2:].split('.')[0]
# 	# print(lrnum,gtnum,depthnum)
# 	if lrnum != gtnum or gtnum != depthnum:
# 		error1 += 1
#
# print(error1)

# 视差图是灰度图，只有一个通道0-255，越近强度越大,越亮
# img1 = Image.open('./data/train/depth/1.jpg')
# array = np.asarray(img1)
# data = t.from_numpy(array)

# array3 = np.ones((array.shape))
# array3[(array3.shape[0]-30):(array3.shape[0]+30),(array3.shape[1]-30):(array3.shape[1]+30)] = 0
# plt.imshow(array3,cmap="gray")
# plt.show()

# print(array.shape)
# print(array)
# print(array.max())
# print(array.min())

# GT 灰度图，只有一个通道(只有true、fasle)
# img2 = Image.open('./data/train/GT/1.png')
# array2 = np.asarray(img2)
# print(array2.shape)
# print(array2)
# print(array2.max())
# print(array2.min())


# imgcarsaliency = Image.open('../carsaliency.jpeg')
# arraysali = np.asarray(imgcarsaliency)
# print(arraysali)
# print(arraysali.max())
# print(arraysali.min())
# print(arraysali[(arraysali.shape[0] //2 -30):(arraysali.shape[0] //2+30),(arraysali.shape[1] //2-30):(arraysali.shape[1] //2+30)])
#
#
# plt.subplot(121)
# plt.imshow(arraysali,cmap='gray')
# plt.axis('off')
#
# plt.subplot(122)
# plt.imshow(arraysali / 255,cmap='gray')
# plt.axis('off')
#
# plt.show()


class NJUDateset(Dataset):
    def __init__(self, train,  transform=None):
        self.train = train
        # if self.train:
        self.lrimgs = lrimgs
        self.depth = depth
        self.gt = gt
        self.bound = bound
        # else:
        #     self.lrimgs = RGBval
        #     self.depth = depthval
        #     self.gt = gtval
        #     self.bound = boundval
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        depthPath = self.depth[index]
        gtPath = self.gt[index]
        boundPath = self.bound[index]
        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        depth = np.asarray(depth)  # 0,255
        gt = Image.open(gtPath)
        gt = np.asarray(gt).astype(np.float)

        if gt.max() == 255.:
            gt = gt / 255.
        gt_b = Image.open(boundPath)
        gt_b = np.asarray(gt_b).astype(np.float)
        if gt_b.max() == 255.:
            gt_b = gt_b / 255.
        # name = imgPath.split('/')[-1].split('.')[-2]

        sample = {'image': img, 'depth': depth, 'label': gt, 'bound': gt_b}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)


class Test(Dataset):
    def __init__(self, transform=None):
        self.lrimgs = RGBVT5000Test
        self.depth = depthVT5000Test
        self.gt = gtVT5000Test
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.lrimgs[index]
        # print(imgPath)
        depthPath = self.depth[index]
        gtPath = self.gt[index]
        # bound = self.bound[index]

        img = Image.open(imgPath)  # 0到255
        img = np.asarray(img)
        depth = Image.open(depthPath)  # 0到255   直接是深度信息过的话就是原来的深度信息大小
        depth = np.asarray(depth)  # 0,255
        gt = Image.open(gtPath)
        gt = np.asarray(gt).astype(np.float)
        name = imgPath.split('/')[-1].split('.')[-2]
        if gt.max() == 255.:
            gt = gt / 255.
        # gt_b = Image.open(bound)
        # gt_b = np.asarray(gt_b).astype(np.float)
        # if gt_b.max() == 255.:
        #     gt_b = gt_b / 255.

        sample = {'image': img, 'depth': depth, 'label': gt,'name':name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.lrimgs)


trainData = NJUDateset(train=True, transform=transforms.Compose([
    data_transform.scaleNorm(),
    data_transform.RandomHSV((0.9,1.1),(0.9,1.1),(25,25)),
    data_transform.RandomFlip(),
    data_transform.ToTensor(),
    data_transform.Normalize()
]))

# valData = NJUDateset(train=False, transform=transforms.Compose([
#     data_transform.scaleNorm(),
#     # data_transform.RandomHSV((0.9,1.1),(0.9,1.1),(25,25)),
#     # data_transform.RandomFlip(),
#     data_transform.ToTensor(),
#     data_transform.Normalize()
# ]
# ))

testData = Test(transform=transforms.Compose([
    data_transform.scaleNorm(),
    # data_transform.RandomHSV((0.9,1.1),(0.9,1.1),(25,25)),
    # data_transform.RandomFlip(),
    data_transform.ToTensor(),
    data_transform.Normalize()
]
))


if __name__ == '__main__':
    sample = testData[442]
    l1 = sample['label']
    l2 = sample['label2']
    l3 = sample['label3']
    l4 = sample['label4']
    l5 = sample['label5']
    img = sample['image']
    depth = sample['depth']
    # bound = sample['bound']
    name = sample['name']
    print(l1.size())
    print(name)
    # print(img)
    # print(np.max(depth))
    # print(img.shape)
    # print(depth.shape)
    # print(bound.shape)
    # import numpy as np
    # uni1 = np.unique(l1)
    # print(uni1)
    # uni1 = np.unique(l2)
    # print(uni1)
    # uni1 = np.unique(l3)
    # print(uni1)
    # uni1 = np.unique(l4)
    # print(uni1)
    # uni1 = np.unique(l5)
    # print(uni1)
    # bound = np.unique(bound)
    # print(bound)

# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# train_dataloader = DataLoader(trainData,batch_size=2,shuffle=True,num_workers=4)
# for i,sample in enumerate(train_dataloader):
# 	image = Variable(sample['image'])
# 	depth = Variable(sample['depth'])
# 	label = Variable(sample['label'].long())
# 	label2 = Variable(sample['label2'].long())
# 	label3 = Variable(sample['label3'].long())
# 	label4 = Variable(sample['label4'].long())
# 	label5 = Variable(sample['label5'].long())
# 	a = label2.data.numpy()
# 	b = a[0]
#
# 	plt.imshow(b,cmap='gray')
# 	plt.show()


# img = testData[1]['image']
# dep = testData[1]['depth']
# lab = testData[1]['label']
# print(lab.flatten())
# num = lab.numpy()
# print(dep)
# print(img)
# u = np.unique(num,np.long)
#
# print(u)


# import matplotlib.pyplot as plt
# for i in range(100):
# 	img = trainData[i]['image']
# 	depth = trainData[i]['depth']
# 	label = trainData[i]['label']
# 	img = img.numpy().transpose((1,2,0))
# 	depth = depth[0].numpy()
# 	label = label.squeeze().numpy()
# 	_,figs = plt.subplots(3,1,figsize=(12,10))
# 	figs[0].imshow(img)
# 	figs[1].imshow(depth,cmap='gray')
# 	figs[2].imshow(label,cmap='gray')
# 	plt.show()
