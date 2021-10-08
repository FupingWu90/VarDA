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
from utils_for_transfer import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)


EPOCH = 30
KLDLamda=1.0

# PredLamda=1e3
# DisLamda=1e-4
LR = 1e-3
ADA_DisLR=1e-4

WEIGHT_DECAY =1e-5
WORKERSNUM = 10
#TestDir=['/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192_Validation/','/home/wfp/2019TMI/LGE_C0_T2/Original/c0t2lgeCropNorm/LGE192/']
prefix='~/experiments/loss_tSNE'
dataset_dir = '~/Dataset/Patch192'
TestDir=dataset_dir+'/LGE_Vali/'
BatchSize = 10
KERNEL=4


# SAVE_DIR =prefix+ '/save_train_param'
# SAVE_IMG_DIR=prefix+'/save_test_label'

def ADA_Train(source_vae_loss_list,source_seg_loss_list,target_vae_loss_list,distance_loss_list, Train_LoaderA,Train_LoaderB,encoder,decoderA,decoderAdown2,decoderAdown4,decoderB,decoderBdown2,decoderBdown4,gate,DistanceNet,lr,kldlamda,predlamda,alpha,beta,infolamda,epoch,optim, savedir):
    lr=lr*(0.9**(epoch))
    for param_group in optim.param_groups:
        param_group['lr'] = lr


    A_iter = iter(Train_LoaderA)
    B_iter = iter(Train_LoaderB)

    i=0


    while i<len(A_iter)-1 and i<len(B_iter)-1:
        ct,ct_down2,ct_down4,label,label_down2,label_down4 ,info_ct= A_iter.next()
        mr,mr_down2,mr_down4,info_mr= B_iter.next()

        ct= ct.cuda()
        ct_down2= ct_down2.cuda()
        ct_down4= ct_down4.cuda()
        #info_ct = info_ct.cuda()

        mr= mr.cuda()
        mr_down4= mr_down4.cuda()
        mr_down2= mr_down2.cuda()
        #info_mr = info_mr.cuda()

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
        #info_pred_ct = Infonet(info_pred_ct)

        #info_cri = nn.CrossEntropyLoss().cuda()
        #infoloss_ct = info_cri(info_pred_ct,info_ct)

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
        #info_pred_mr = Infonet(info_pred_mr)

        #infoloss_mr = info_cri(info_pred_mr,info_mr)

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


        source_loss = 10.0*BCE_ct+kldlamda*KLD_ct+predlamda*(segloss_output+fusionsegloss_output)+10.0*BCE_down2_ct + kldlamda*KLD_down2_ct +predlamda * segdown2loss_output+10.0*BCE_down4_ct + kldlamda*KLD_down4_ct +predlamda * segdown4loss_output#+ infolamda*infoloss_ct
        target_loss = 10.0*BCE_mr+kldlamda*KLD_mr+10.0*BCE_down2_mr + kldlamda*KLD_down2_mr +10.0*BCE_down4_mr + kldlamda*KLD_down4_mr#+infolamda*infoloss_mr
        discrepancy_loss = distance_loss+distance_down2_loss + 1e-1*distance_down4_loss


        source_vae_loss_list.append(1*(BCE_ct+BCE_down2_ct+BCE_down4_ct+KLD_ct+KLD_down2_ct+KLD_down4_ct).item())
        source_seg_loss_list.append(1 * (segloss_output+fusionsegloss_output+segdown2loss_output+segdown4loss_output).item())
        target_vae_loss_list.append(1 * (BCE_mr+BCE_down2_mr+BCE_down4_mr+KLD_mr+KLD_down2_mr+KLD_down4_mr).item())
        distance_loss_list.append(1e-5 * discrepancy_loss.item())

        balanced_loss = source_loss+alpha*target_loss+beta*discrepancy_loss

        optim.zero_grad()
        balanced_loss.backward()
        optim.step()

        if i % 20 == 0:
            print('epoch %d , %d th iter; seglr,ADA_totalloss,segloss,distance_loss1,distance_loss2: %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f'\
                  % (epoch, i,lr, balanced_loss.item(),BCE_mr.item(),KLD_mr.item(),BCE_ct.item(),KLD_ct.item(),fusionsegloss_output.item(),segloss_output.item(),segdown2loss_output.item(),segdown4loss_output.item(),distance_loss.item(),distance_down2_loss.item(),distance_down4_loss.item()))

        i=i+1

def SegNet_test_mr(dir, mrSegNet, gate,epoch,ePOCH, save_DIR):
    criterion=0

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
        npimg = sitk.GetArrayFromImage(itkimg)
        npimg = npimg.astype(np.float32)


        data=torch.from_numpy(np.expand_dims(npimg,axis=1)).type(dtype=torch.FloatTensor).cuda()

        label=torch.from_numpy(nplab).cuda()

        truearg  = np.zeros((data.size(0),data.size(2),data.size(3)))

        for slice in range(data.size(0)):
            output,_,_, _, _, _ ,_,_,_,_,_,_,_,_,_,_,_= mrSegNet(data[slice:slice+1,:,:,:], gate)

            truemax, truearg0 = torch.max(output, 1, keepdim=False)
            truearg0 = truearg0.detach().cpu().numpy()
            truearg[slice:slice+1,:,:]=truearg0


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


def t_SNE_plot(Train_LoaderA,Train_LoaderB,net,save_dir,mode):
    A_iter = iter(Train_LoaderA)
    print (len(A_iter))
    B_iter = iter(Train_LoaderB)
    print (len(B_iter))
    net.eval()

    features_A = np.zeros((64,))
    features_B = np.zeros((64,))
    print ('begin init')




    i = 0

    while i < len(A_iter)-1 and i < len(B_iter)-1:
        ct, ct_down2, ct_down4, label, label_down2, label_down4, info_ct = A_iter.next()
        mr, mr_down2, mr_down4, info_mr = B_iter.next()

        ct = ct.cuda()

        mr = mr.cuda()

        _, _, _, feat_ct, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(ct, 0.0)

        _, _, _, feat_mr, _, _, _, _, _, _, _, _, _, _, _, _, _  = net(mr, 0.0)

        features_A = np.vstack((features_A, feat_ct.cpu().detach().numpy().mean(axis=(2,3)).reshape(ct.size(0),-1)))
        features_B = np.vstack((features_B, feat_mr.cpu().detach().numpy().mean(axis=(2,3)).reshape(mr.size(0),-1)))
        i=i+1

        print (i)


    tsne = TSNE()
    print ('tsne class')
    print (features_A.shape)
    X_embedded = tsne.fit_transform(np.concatenate((features_A[1:],features_B[1:]),axis=0))
    print ('finish mapping')
    Y = ['source']*features_A[1:].shape[0]+['target']*features_A[1:].shape[0]
    #Y = ['source'] * 500 + ['target'] * 500

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=Y, legend='full', palette=palette)

    plt.savefig(os.path.join(save_dir, '{}.png'.format(mode)))
    plt.close()

    np.save(os.path.join(save_dir, '{}_X.npy'.format(mode)), X_embedded)
    np.save(os.path.join(save_dir, '{}_Y.npy'.format(mode)), np.array(Y))




def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    cudnn.benchmark = True
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

    #Infonet = InfoNet().cuda()

    DistanceNet = Gaussian_Distance(KERNEL)  #64,Num_Feature2,(12,12)
    DistanceNet = DistanceNet.cuda()
    #DistanceNet2 = nn.DataParallel(DistanceNet2, device_ids=[0,1])


    DA_optim = torch.optim.Adam([{'params': vaeencoder.parameters()},{'params': source_vaedecoder.parameters()},{'params': source_down2_vaedecoder.parameters()},{'params': source_down4_vaedecoder.parameters()},{'params': target_vaedecoder.parameters()},{'params': target_down2_vaedecoder.parameters()},{'params': target_down4_vaedecoder.parameters()}],lr=LR,weight_decay=WEIGHT_DECAY)

    SourceData = C0_TrainSet(dataset_dir)
    SourceData_loader = DataLoader(SourceData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)

    TargetData = LGE_TrainSet(dataset_dir)
    TargetData_loader = DataLoader(TargetData, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)


    # TestData = LabeledDataSet(modality='mr',stage='test')
    # TestData_loader = DataLoader(TestData, batch_size=1, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)
    PredLamda=1e3
    # DisLamdaList=[1e-3,1e-4]
    # DisLamdaListDown2=[1e-3,1e-4]
    # DisLamdaListDown4=[1e-3,1e-4]
    #InfoLamda=1e2
    Alpha=1e0
    Beta=1e-3

    SAVE_DIR=prefix+'/save_param'+str(Beta)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    vaeencoder.apply(init_weights)
    source_vaedecoder.apply(init_weights)
    source_down2_vaedecoder.apply(init_weights)
    source_down4_vaedecoder.apply(init_weights)
    target_vaedecoder.apply(init_weights)
    target_down2_vaedecoder.apply(init_weights)
    target_down4_vaedecoder.apply(init_weights)

    source_vae_loss_list=[]
    source_seg_loss_list = []
    target_vae_loss_list=[]
    distance_loss_list=[]
    print ('start init tsne')

    t_SNE_plot(SourceData_loader, TargetData_loader, vaeencoder, SAVE_DIR, 'init_tsne')

    print ('finish init tsne')


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
        ADA_Train(source_vae_loss_list,source_seg_loss_list,target_vae_loss_list,distance_loss_list, SourceData_loader,TargetData_loader,vaeencoder,source_vaedecoder,source_down2_vaedecoder,source_down4_vaedecoder,target_vaedecoder,target_down2_vaedecoder,target_down4_vaedecoder,1.0,DistanceNet,LR,KLDLamda,PredLamda,Alpha,Beta,InfoLamda,epoch,DA_optim, SAVE_DIR)
        vaeencoder.eval()
        criter =SegNet_test_mr(TestDir, vaeencoder,0, epoch,EPOCH, SAVE_DIR)
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

    print ('best epoch:%d' % (best_epoch))
    with open("%s/lge_testout_index.txt" % (SAVE_DIR), "a") as f:
        f.writelines(["\n\nbest epoch:%d, iter num:%d" % (best_epoch,len(source_vae_loss_list))])



    ## plot loss
    print ('\n iter num: %d' % (len(source_vae_loss_list)))

    source_vae_loss_list=np.array(source_vae_loss_list)
    source_seg_loss_list = np.array(source_seg_loss_list)
    target_vae_loss_list=np.array(target_vae_loss_list)
    distance_loss_list=np.array(distance_loss_list)
    np.save(os.path.join(SAVE_DIR, 'source_bce_loss_list.npy'), source_vae_loss_list)
    np.save(os.path.join(SAVE_DIR, 'source_seg_loss_list.npy'), source_seg_loss_list)
    np.save(os.path.join(SAVE_DIR, 'target_bce_loss_list.npy'), target_vae_loss_list)
    np.save(os.path.join(SAVE_DIR, 'distance_loss_list.npy'), distance_loss_list)


    plt.plot(np.arange(0, source_vae_loss_list.shape[0]),
             source_vae_loss_list, 'r', linestyle="-",
             label=r'$\widetilde{\mathcal{L}}_{S/seg}$')
    plt.plot(np.arange(0, source_seg_loss_list.shape[0]), source_seg_loss_list,
             'b',
             label=r'$\widetilde{\mathcal{L}}_{S:seg}$')
    plt.plot(np.arange(0, target_vae_loss_list.shape[0]),
             target_vae_loss_list, 'g', linestyle="-",
             label=r'$\widetilde{\mathcal{L}}_{T}$')
    plt.plot(np.arange(0, distance_loss_list.shape[0]), distance_loss_list,
             'm',
             label=r'$\widetilde{\mathcal{D}}$')

    plt.legend()
    #xticks = [0, 3000, 6000, 9000, 12000, 15000]
    #xtick_labels = ['0', '3k', '6k', '9k', '12k', '15k']
    #plt.xticks(xticks, xtick_labels, fontsize=8)
    #plt.xlim([0, 15000])
    # plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    # plt.register_cmap(cmap=viridis)
    # plt.set_cmap(cmaps.viridis)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    plt.xlabel('iterations', fontsize=15)
    plt.ylabel(" Losses", fontsize=15)
    plt.grid(axis="y", linestyle='--')
    plt.margins(x=0)
    plt.savefig(os.path.join(SAVE_DIR, 'loss.png'))
    plt.close()

    vaeencoder.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'encoder_param.pkl')))
    t_SNE_plot(SourceData_loader, TargetData_loader, vaeencoder, SAVE_DIR, 'da_tsne')


if __name__ == '__main__':
    main()
