import torch
from torch import nn
from torch.utils.data import Dataset
import os
import math
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import time
import scipy.misc
from utils_for_transfer import *

EPOCH = 30
KLDLamda=1.0

# PredLamda=1e3
# DisLamda=1e-4
LR = 1e-4
ADA_DisLR=1e-4

WEIGHT_DECAY =1e-5
WORKERSNUM = 10
dataset_dir = '/home/wfp/2021TMI_Rebuttal/Dataset/Patch192'
prefix='/home/wfp/2021TMI_Rebuttal/experiments/target_num'
#TestDir=['/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192_Validation/','/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192/']
TestDir=[dataset_dir+'/LGE_Test/',dataset_dir+'/LGE_Vali/']
BatchSize = 8
KERNEL=4

# SAVE_DIR =prefix+ '/save_train_param'
# SAVE_IMG_DIR=prefix+'/save_test_label'

def ADA_Train( Train_LoaderA,Train_LoaderB,Infonet,encoder,decoderA,decoderAdown2,decoderAdown4,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,kldlamda,predlamda,dislamda,dislamdadown2,dislamdadown4,infolamda,epoch,optim, savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0


    while i<len(A_iter) and i<len(B_iter):
        ct,ct_down2,ct_down4,label,label_down2,label_down4 ,info_ct= A_iter.next()
        mr,mr_down2,mr_down4,info_mr= B_iter.next()

        ct= ct.cuda()
        ct_down2= ct_down2.cuda()
        ct_down4= ct_down4.cuda()
        info_ct = info_ct.cuda()

        mr= mr.cuda()
        mr_down4= mr_down4.cuda()
        mr_down2= mr_down2.cuda()
        info_mr = info_mr.cuda()

        label= label.cuda()
        label_onehot =torch.FloatTensor(label.size(0), 4,label.size(1),label.size(2)).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, label.unsqueeze(dim=1), 1)

        label_down2= label_down2.cuda()
        label_down2_onehot =torch.FloatTensor(label_down2.size(0), 4,label_down2.size(1),label_down2.size(2)).cuda()
        label_down2_onehot.zero_()
        label_down2_onehot.scatter_(1, label_down2.unsqueeze(dim=1), 1)

        label_down4= label_down4.cuda()
        label_down4_onehot =torch.FloatTensor(label_down4.size(0), 4,label_down4.size(1),label_down4.size(2)).cuda()
        label_down4_onehot.zero_()
        label_down4_onehot.scatter_(1, label_down4.unsqueeze(dim=1), 1)

        fusionseg,_, out_ct,feat_ct, mu_ct,logvar_ct, _, outdown2_ct,featdown2_ct, mudown2_ct,logvardown2_ct,_, outdown4_ct,featdown4_ct, mudown4_ct,logvardown4_ct,info_pred_ct= encoder(ct,gate)
        info_pred_ct = Infonet(info_pred_ct)

        info_cri = nn.CrossEntropyLoss().cuda()
        infoloss_ct = info_cri(info_pred_ct,info_ct)

        seg_criterian = BalancedBCELoss(label)
        seg_criterian = seg_criterian.cuda()
        segloss_output = seg_criterian(out_ct, label)
        fusionsegloss_output = seg_criterian(fusionseg, label)

        segdown2_criterian = BalancedBCELoss(label_down2)
        segdown2_criterian = segdown2_criterian.cuda()
        segdown2loss_output = segdown2_criterian(outdown2_ct, label_down2)

        segdown4_criterian = BalancedBCELoss(label_down4)
        segdown4_criterian = segdown4_criterian.cuda()
        segdown4loss_output = segdown4_criterian(outdown4_ct, label_down4)

        recon_ct=decoderA(feat_ct,label_onehot)
        BCE_ct = F.binary_cross_entropy(recon_ct, ct)
        KLD_ct = -0.5 * torch.mean(1 + logvar_ct - mu_ct.pow(2) - logvar_ct.exp())

        recondown2_ct=decoderAdown2(featdown2_ct,label_down2_onehot)
        BCE_down2_ct = F.binary_cross_entropy(recondown2_ct, ct_down2)
        KLD_down2_ct = -0.5 * torch.mean(1 + logvardown2_ct - mudown2_ct.pow(2) - logvardown2_ct.exp())

        recondown4_ct=decoderAdown4(featdown4_ct,label_down4_onehot)
        BCE_down4_ct = F.binary_cross_entropy(recondown4_ct, ct_down4)
        KLD_down4_ct = -0.5 * torch.mean(1 + logvardown4_ct - mudown4_ct.pow(2) - logvardown4_ct.exp())

        _,pred_mr, _,feat_mr, mu_mr,logvar_mr, preddown2_mr, _,featdown2_mr, mudown2_mr,logvardown2_mr,preddown4_mr, _,featdown4_mr, mudown4_mr,logvardown4_mr,info_pred_mr= encoder(mr,gate)
        info_pred_mr = Infonet(info_pred_mr)

        infoloss_mr = info_cri(info_pred_mr,info_mr)

        recon_mr=decoderB(feat_mr,pred_mr)
        BCE_mr = F.binary_cross_entropy(recon_mr, mr)
        KLD_mr = -0.5 * torch.mean(1 + logvar_mr - mu_mr.pow(2) - logvar_mr.exp())

        recondown2_mr=decoderBdown2(featdown2_mr,preddown2_mr)
        BCE_down2_mr = F.binary_cross_entropy(recondown2_mr, mr_down2)
        KLD_down2_mr = -0.5 * torch.mean(1 + logvardown2_mr - mudown2_mr.pow(2) - logvardown2_mr.exp())

        recondown4_mr=decoderBdown4(featdown4_mr,preddown4_mr)
        BCE_down4_mr = F.binary_cross_entropy(recondown4_mr, mr_down4)
        KLD_down4_mr = -0.5 * torch.mean(1 + logvardown4_mr - mudown4_mr.pow(2) - logvardown4_mr.exp())

        distance_loss = DistanceNet(mu_ct,logvar_ct,mu_mr,logvar_mr)
        distance_down2_loss = DistanceNet(mudown2_ct,logvardown2_ct,mudown2_mr,logvardown2_mr)
        distance_down4_loss = DistanceNet(mudown4_ct,logvardown4_ct,mudown4_mr,logvardown4_mr)


        balanced_loss = 10.0*BCE_mr+torch.mul(KLD_mr,kldlamda)+10.0*BCE_ct+torch.mul(KLD_ct,kldlamda)+torch.mul(distance_loss,dislamda)+predlamda*(segloss_output+fusionsegloss_output)+ \
                        10.0*BCE_down2_ct + torch.mul(KLD_down2_ct, kldlamda) + 10.0*BCE_down2_mr + torch.mul(KLD_down2_mr, kldlamda) + torch.mul(distance_down2_loss, dislamdadown2) + predlamda * segdown2loss_output+ \
                        10.0*BCE_down4_ct + torch.mul(KLD_down4_ct, kldlamda) + 10.0*BCE_down4_mr + torch.mul(KLD_down4_mr, kldlamda) + torch.mul(distance_down4_loss, dislamdadown4) + predlamda * segdown4loss_output+infolamda*(infoloss_mr+infoloss_ct)

        optim.zero_grad()
        balanced_loss.backward()
        optim.step()

        if i % 20 == 0:
            print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
                  % (epoch, i,lr, balanced_loss.item(),BCE_mr.item(),KLD_mr.item(),BCE_ct.item(),KLD_ct.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item(),infoloss_ct.item(),infoloss_mr.item()))

        i=i+1

def SegNet_test_mr(test_dir, mrSegNet, gate,epoch,ePOCH, save_DIR,save_IMG_DIR):
    criterion=0
    for dir in test_dir:
        labsname = glob.glob(dir + '*manual.nii*')

        total_dice = np.zeros((4,))
        total_Iou = np.zeros((4,))

        total_overlap =np.zeros((1,4, 5))
        total_surface_distance=np.zeros((1,4, 5))

        num = 0
        mrSegNet.eval()
        for i in range(len(labsname)):
            itklab = sitk.ReadImage(labsname[i])
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            imgname = labsname[i].replace('_manual.nii', '.nii')
            itkimg = sitk.ReadImage(imgname)
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1
            npimg = npimg.astype(np.float32)


            # data = np.transpose(
            #     transform.resize(np.transpose(npimg, (1, 2, 0)), (96, 96),
            #                      order=3, mode='edge', preserve_range=True), (2, 0, 1))
            data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).cuda()

            label=torch.from_numpy(nplab).cuda()

            truearg  = np.zeros((data.size(0),data.size(2),data.size(3)))

            for slice in range(data.size(0)):
                output,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= mrSegNet(data[slice:slice+1,:,:,:], gate)

                truemax, truearg0 = torch.max(output, 1, keepdim=False)
                truearg0 = truearg0.detach().cpu().numpy()
                truearg[slice:slice+1,:,:]=truearg0
            #truearg = np.transpose(transform.resize(np

            #
            # truemax, truearg = torch.max(output, 1, keepdim=False)
            # truearg = truearg.detach().cpu().numpy()
            # truearg = np.transpose(transform.resize(np.transpose(truearg, (1, 2, 0)), (192,192), order=0,mode='edge', preserve_range=True), (2, 0, 1)).astype(np.int64)

            dice = dice_compute(truearg,label.cpu().numpy())
            Iou = IOU_compute(truearg,label.cpu().numpy())
            overlap_result, surface_distance_result = Hausdorff_compute(truearg,label.cpu().numpy(),itkimg.GetSpacing())

            total_dice = np.vstack((total_dice,dice))
            total_Iou = np.vstack((total_Iou,Iou))

            total_overlap = np.concatenate((total_overlap,overlap_result),axis=0)
            total_surface_distance = np.concatenate((total_surface_distance,surface_distance_result),axis=0)

            num+=1

        if num==0:
            return
        else:
            meanDice = np.mean(total_dice[1:],axis=0)
            stdDice = np.std(total_dice[1:],axis=0)

            meanIou = np.mean(total_Iou[1:],axis=0)
            stdIou = np.std(total_Iou[1:],axis=0)

            mean_overlap = np.mean(total_overlap[1:], axis=0)
            std_overlap = np.std(total_overlap[1:], axis=0)

            mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
            std_surface_distance = np.std(total_surface_distance[1:], axis=0)

            if 'Vali' in dir:
                phase='validate'
            else:
                criterion = np.mean(meanDice[1:])
                phase='test'
            with open("%s/lge_testout_index.txt" % (save_DIR), "a") as f:
                f.writelines(["\n\nepoch:", str(epoch), " ",phase," ", "\n","meanDice:",""\
                                 ,str(meanDice.tolist()),"stdDice:","",str(stdDice.tolist()),"","\n","meanIou:","",str(meanIou.tolist()),"stdIou:","",str(stdIou.tolist()), \
                                  "", "\n\n","jaccard, dice, volume_similarity, false_negative, false_positive:", "\n","mean:", str(mean_overlap.tolist()),"\n", "std:", "", str(std_overlap.tolist()), \
                                  "", "\n\n","hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:", "\n","mean:", str(mean_surface_distance.tolist()), "\n","std:", str(std_surface_distance.tolist())])
    return criterion




def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    cudnn.benchmark = True

    PredLamda=1e3
    DisLamda=1e-3
    DisLamdaDown2=1e-3
    DisLamdaDown4=1e-4
    InfoLamda=1e2

    sample_nums = [45]

    for sample_num in sample_nums:

        SAVE_DIR=prefix+'/save_train_param'+'_num'+str(sample_num)
        SAVE_IMG_DIR=prefix+'/save_test_label'+'_num'+str(sample_num)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.mkdir(SAVE_IMG_DIR)

        vaeencoder = VAE()
        vaeencoder = vaeencoder.cuda()

        source_vaedecoder = VAEDecode()
        source_vaedecoder = source_vaedecoder.cuda()

        source_down2_vaedecoder = VAEDecode_down2()
        source_down2_vaedecoder = source_down2_vaedecoder.cuda()

        source_down4_vaedecoder = VAEDecode_down4()
        source_down4_vaedecoder = source_down4_vaedecoder.cuda()

        target_vaedecoder = VAEDecode()
        target_vaedecoder = target_vaedecoder.cuda()

        target_down2_vaedecoder = VAEDecode_down2()
        target_down2_vaedecoder = target_down2_vaedecoder.cuda()

        target_down4_vaedecoder = VAEDecode_down4()
        target_down4_vaedecoder = target_down4_vaedecoder.cuda()

        Infonet = InfoNet().cuda()

        DistanceNet = Gaussian_Distance(KERNEL)  # 64,Num_Feature2,(12,12)
        DistanceNet = DistanceNet.cuda()
        # DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])


        DA_optim = torch.optim.Adam([{'params': Infonet.parameters()}, {'params': vaeencoder.parameters()},
                                     {'params': source_vaedecoder.parameters()},
                                     {'params': source_down2_vaedecoder.parameters()},
                                     {'params': source_down4_vaedecoder.parameters()},
                                     {'params': target_vaedecoder.parameters()},
                                     {'params': target_down2_vaedecoder.parameters()},
                                     {'params': target_down4_vaedecoder.parameters()}], lr=LR,
                                    weight_decay=WEIGHT_DECAY)

        SourceData = C0_TrainSet(dataset_dir,35)
        SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,
                                       pin_memory=True)

        TargetData = LGE_TrainSet(dataset_dir,sample_num)
        TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,
                                       pin_memory=True)

        vaeencoder.apply(init_weights)
        source_vaedecoder.apply(init_weights)
        source_down2_vaedecoder.apply(init_weights)
        source_down4_vaedecoder.apply(init_weights)
        target_vaedecoder.apply(init_weights)
        target_down2_vaedecoder.apply(init_weights)
        target_down4_vaedecoder.apply(init_weights)

        criterion=0
        best_epoch=0
        for epoch in range(EPOCH):
            vaeencoder.train()
            source_vaedecoder.train()
            source_down2_vaedecoder.train()
            source_down4_vaedecoder.train()
            target_vaedecoder.train()
            target_down2_vaedecoder.train()
            target_down4_vaedecoder.train()
            ADA_Train( SourceData_loader,TargetData_loader,Infonet,vaeencoder,source_vaedecoder,source_down2_vaedecoder,source_down4_vaedecoder,target_vaedecoder,target_down2_vaedecoder,target_down4_vaedecoder,1.0,DistanceNet,LR,KLDLamda,PredLamda,DisLamda,DisLamdaDown2,DisLamdaDown4,InfoLamda,epoch,DA_optim, SAVE_DIR)
            vaeencoder.eval()
            criter =SegNet_test_mr(TestDir, vaeencoder,0, epoch,EPOCH, SAVE_DIR,SAVE_IMG_DIR)
            if criter > criterion:
                best_epoch = epoch
                criterion=criter
                torch.save(vaeencoder.state_dict(), os.path.join(SAVE_DIR, 'encoder_param.pkl'))
                torch.save(source_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderA_param.pkl'))
                torch.save(source_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown2_param.pkl'))
                torch.save(source_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderAdown4_param.pkl'))
                torch.save(target_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderB_param.pkl'))
                torch.save(target_down2_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown2_param.pkl'))
                torch.save(target_down4_vaedecoder.state_dict(), os.path.join(SAVE_DIR, 'decoderBdown4_param.pkl'))
        print ('\n')
        print ('\n')
        print ('best epoch:%d' % (best_epoch))
        with open("%s/lge_testout_index.txt" % (SAVE_DIR), "a") as f:
            f.writelines(["\n\nbest epoch:%d" % (best_epoch)])


        del vaeencoder, source_vaedecoder, source_down2_vaedecoder, source_down4_vaedecoder, target_vaedecoder, target_down2_vaedecoder, target_down4_vaedecoder


if __name__ == '__main__':
    main()
