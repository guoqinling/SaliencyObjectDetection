import torch as t
from torch import nn
import numpy as np
from RGBT.dataprocessing_rgbt import trainData
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime
import os
from torch.autograd import Variable
import torch.nn.functional as F
# from paper5_model.DFNet_vgg8 import DFNet
# from RGBT_net.Twice import Twice
from letters.newxr4 import my64
# from letters.xr3 import my64
# from pytorch_ssim import *
# from pytorch_iou import *
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
from scipy import misc
from ranger import Ranger
# from variety_loss import BCELOSS

class BCELOSS(nn.Module):  # 定义二分类交叉熵（Binary Cross Entropy）loss
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, target_scale):
        losses = []
        for inputs, targets in zip(input_scale, target_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

# writer = SummaryWriter(os.path.join('../train_model/results'))


batchsize = 4

train_dataloader = DataLoader(trainData, batch_size=batchsize, shuffle=True, num_workers=4)

net = my64()
net.cuda()

# criterion_mix = MIXBCELOSS().cuda()
criterion_normal_log = nn.BCEWithLogitsLoss().cuda()
criterion_normal = BCELOSS().cuda()
criterion_CE = nn.CrossEntropyLoss().cuda()
# criterion_edge = EdgeLoss().cuda()
criterion_test = nn.BCELoss().cuda()
# criterion_edge_loss = EdgeLoss_wjw().cuda()
# criterion_CMMFloss = CMM_Floss().cuda()
# criterion_cat = TooMuchLoss().cuda()
# criterion_no_local = Loss().cuda()


# optimizer = Ranger(net.parameters(), lr=5e-5, weight_decay=1e-3)
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)
best = [10000]
bgbest = [10000]
maelist = [10000]
numloss = 0
nummae = 0
trainloss = []
testloss = []
maeloss = []
b = 0.5


for epoch  in range(150):  # 动态修改学习率
    if epoch % 50 == 0 and epoch != 0:
        for group in optimizer.param_groups:
            group['lr'] *= 0.5
    train_loss = 0
    net = net.train()
    prec_time = datetime.now()
    maeval = 0

    for i, sample in enumerate(train_dataloader):
        image = Variable(sample['image'].cuda())
        depth = Variable(sample['depth'].cuda())
        batch_num = sample['image'].size()[0]
        image1 = Variable(t.FloatTensor(np.ones(batch_num, dtype=float)), requires_grad=False).cuda()
        depth1 = Variable(t.FloatTensor(np.zeros(batch_num, dtype=float)), requires_grad=False).cuda()
        # bound = Variable(sample['bound'].float().cuda())
        # contour = Variable(sample['contour'].float().cuda())
        # rlabel = Variable(sample['rlabel'].float().cuda())
        label = Variable(sample['label'].float().cuda())
        label3 = Variable(sample['label3'].float().cuda())
        label4 = Variable(sample['label4'].float().cuda())
        # bound3 = Variable(sample['bound3'].float().cuda())
        optimizer.zero_grad()
        out = net(image,depth)
        # out = t.sigmoid(out)
        out_1 = t.sigmoid(out[0])
        out_2 = t.sigmoid(out[1])
        out_3 = t.sigmoid(out[2])
        out_4 = t.sigmoid(out[3])
        out_5 = t.sigmoid(out[4])

        # d_out_rgb = out[3]
        # d_out_dep = out[4]
        # D_loss = criterion_test(d_out_rgb, image1.squeeze()) + criterion_test(d_out_dep, depth1.squeeze())
        # loss = criterion_normal(out, label)
        loss_1 = criterion_test(out_1, label)
        loss_2 = criterion_test(out_2, label)
        loss_3 = criterion_test(out_3, label)
        loss_4 = criterion_test(out_4, label)
        loss_5 = criterion_test(out_5, label)

        mae = t.sum(t.abs(label - out_1)) / t.numel(label)
        loss = loss_1+loss_2+loss_3+loss_4+loss_5
        # loss = loss_1
        # loss = (loss-b).abs()+b
        # print('background loss is', i, '========', loss.item())
        print('the total loss is', i, '========', loss.item())
        loss.backward()
        optimizer.step()
        train_loss = loss.item() + train_loss
        maeval = maeval + mae.item()
        niter = epoch * len(train_dataloader) + i

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f},Valid Loss: {:.5f},Valid MAE: {:.5f}'.format(
        epoch, train_loss / len(train_dataloader),
               train_loss / len(train_dataloader),
               maeval / len(train_dataloader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str)
    trainloss.append(train_loss / len(train_dataloader))
    testloss.append(train_loss / len(train_dataloader))
    maeloss.append(maeval / len(train_dataloader))

    if train_loss / len(train_dataloader) <= min(best):
        best.append(train_loss / len(train_dataloader))
        numloss = epoch
        t.save(net.state_dict(), '/media/duanting/shuju/newxr4/'+str(epoch)+'_loss.pth')
    if maeval / len(train_dataloader) <= min(maelist) and epoch > numloss:
        maelist.append(maeval / len(train_dataloader))
        nummae = epoch
        t.save(net.state_dict(), '/media/duanting/shuju/newxr4/'+str(epoch)+'_mae.pth')

    print(best)
    print(maelist)
    print(min(best), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  totalLoss', numloss)
    print(min(maelist), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  MAE", nummae)
    print('--------------------------')

    # np.savetxt('loss_final.txt', trainloss)
    # np.savetxt('loss_final_test.txt', testloss)
    # np.savetxt('loss_final_testmae.txt', maeloss)
