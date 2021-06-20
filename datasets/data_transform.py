import numpy as np
import scipy.io
import imageio
# import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
# import skimage.transform
import random
import torchvision
import torch
import cv2

image_h, image_w = 224, 224


def label2onehot(label):
    h, w = label.shape
    n_labels = int(label.max() + 1)
    one_hot = np.eye(n_labels)[label.astype(np.int32)].reshape(h, w, n_labels)
    one_hot = one_hot.transpose(2, 0, 1)
    # print(one_hot.shape,'-----------------------------------','onehot')
    return one_hot


def lbl_one(label):
    h, w = label.shape
    n_labels = max(label) + 1
    one_hot = np.eye(n_labels)[label]


class RandomHSV(object):
    """
	       Args:
	           h_range (float tuple): random ratio of the hue channel,
	               new_h range from h_range[0]*old_h to h_range[1]*old_h.
	           s_range (float tuple): random ratio of the saturation channel,
	               new_s range from s_range[0]*old_s to s_range[1]*old_s.
	           v_range (int tuple): random bias of the value channel,
	               new_v range from old_v-v_range to old_v+v_range.
	       Notice:
	           h range: 0-1
	           s range: 0-1
	           v range: 0-255
	       """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))

        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        # random.uniform 随机生成一个数字包含最小值，不包含最大值
        h_random = random.uniform(min(self.h_range), max(self.h_range))
        s_random = random.uniform(min(self.s_range), max(self.s_range))
        v_random = random.uniform(min(self.v_range), max(self.v_range))
        # np.clip截断吧小于最小值的设为最小值，大于最大值的设为最大值
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)

        # np 下的图片 H × W × C，所以在第三个唯独叠加
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        # return {'image': img_new, 'depth': sample['depth'], 'label': sample['label'], 'bound': sample['bound']}
        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    '''
	输入图像进行双线性插值的缩小或方法，对深度和标签图进行近邻缩小或方法
	规定尺寸
	'''



    def __call__(self, sample):  # __call__ 定义将类实例变为可调用对象

        # image, depth, label, bound = sample['image'], sample['depth'], sample['label'], sample['bound']
        image, depth, label = sample['image'], sample['depth'], sample['label']
        # image, depth, label, name = sample['image'], sample['depth'], sample['label'], sample['name']
        # Bi-linear
        image = cv2.resize(image, (image_h, image_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label_1 = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        label_2 = cv2.resize(label, (image_h, image_w), interpolation=cv2.INTER_NEAREST)

        # bound = cv2.resize(bound, (image_h, image_w), interpolation=cv2.INTER_NEAREST)
        # return {'image': image, 'depth': depth, 'label': label,  'label_1': label_1, 'label_2': label_2, 'bound': bound}
        # return {'image': image, 'depth': depth, 'label': label,  'label_1': label_1, 'label_2': label_2, 'name':name}
        return {'image': image, 'depth': depth, 'label': label,  'label_1': label_1, 'label_2': label_2}
class RandomScale(object):
    '''
	自定义比例
	'''

    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = cv2.resize(image, (target_height, target_width), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (target_height, target_width), interpolation=cv2.INTER_NEAREST)

        label = cv2.resize(label, (target_height, target_width), interpolation=cv2.INTER_NEAREST)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]

        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


# 随机旋转
class RandomFlip(object):
    def __call__(self, sample):
        # image, depth, label, bound = sample['image'], sample['depth'], sample['label'], sample['bound']
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()
            # bound = np.fliplr(bound).copy()
        # return {'image': image, 'depth': depth, 'label': label, 'bound': bound}
        return {'image': image, 'depth': depth, 'label': label}


class Normalize(object):
    def __call__(self, sample):
        # image, depth, label,bound= sample['image'],sample['depth'],sample['label'],sample['bound']
        image, depth, label= sample['image'],sample['depth'],sample['label']
        # image, depth, label, name = sample['image'], sample['depth'],sample['label'],sample['name']
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        # depth = depth - 363 / (31197 - 363)
        depth = depth / 255
        depth = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(depth)

        label = label
        # bound = bound

        sample['image'] = image
        sample['depth'] = depth
        sample['label'] = label
        # sample['bound'] = bound
        # sample['name'] = name
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, depth, label, bound = sample['image'], sample['depth'], sample['label'], sample['bound']
        image, depth, label= sample['image'], sample['depth'], sample['label']
        # image, depth, label, name = sample['image'], sample['depth'], sample['label'], sample['name']
        # Generate different label scales
		# neihbour interpolation

        # Generate different label scales
        label2 = cv2.resize(label, (image_h // 2, image_w // 2), interpolation=cv2.INTER_NEAREST)

        label3 = cv2.resize(label, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)

        label4 = cv2.resize(label, (image_h // 8, image_w // 8), interpolation=cv2.INTER_NEAREST)

        label5 = cv2.resize(label, (image_h // 16, image_w // 16), interpolation=cv2.INTER_NEAREST)
        label6 = cv2.resize(label, (image_h // 32, image_w // 32), interpolation=cv2.INTER_NEAREST)

        # bound1 = cv2.resize(bound, (image_h // 4, image_w // 4), interpolation=cv2.INTER_NEAREST)
        label = (label.astype(np.float))
        label2 = label2.astype(np.float)
        label3 = label3.astype(np.float)
        label4 = label4.astype(np.float)
        label5 = label5.astype(np.float)
        label6 = label6.astype(np.float)
        # bound = bound.astype(np.float)
        # bound = np.expand_dims(bound, 0)

        label = np.expand_dims(label, 0)
        label2 = np.expand_dims(label2, 0)
        label3 = np.expand_dims(label3, 0)
        label4 = np.expand_dims(label4, 0)
        label5 = np.expand_dims(label5, 0)
        label6 = np.expand_dims(label6, 0)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # depth = np.expand_dims(depth, 0).astype(np.float)
        # depth = np.array([depth,depth,depth])         #转化为3，h，w
        depth = depth.transpose((2, 0, 1))

        return {
            'image': torch.from_numpy(image).float(),
            'depth': torch.from_numpy(depth).float(),
            'label': torch.from_numpy(label).float(),
            'label2': torch.from_numpy(label2).float(),
            'label3': torch.from_numpy(label3).float(),
            'label4': torch.from_numpy(label4).float(),
            'label5': torch.from_numpy(label5).float(),
            'label6': torch.from_numpy(label6).float(),
            # 'bound': torch.from_numpy(bound).float(),
            # 'name': name

        }
