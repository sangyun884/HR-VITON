import torch
import torch.nn as nn

from torchvision.utils import make_grid
from networks import make_grid as mkgrid

import argparse
import os
import time
from cp_dataset import CPDataset, CPDatasetTest, CPDataLoader
from networks import ConditionGenerator, VGGLoss, GANLoss, load_checkpoint, save_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from torch.utils.data import Subset


def iou_metric(y_pred_batch, y_true_batch):
    B = y_pred_batch.shape[0]
    iou = 0
    for i in range(B):
        y_pred = y_pred_batch[i]
        y_true = y_true_batch[i]
        # y_pred is not one-hot, so need to threshold it
        y_pred = y_pred > 0.5
        
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

    
        intersection = torch.sum(y_pred[y_true == 1])
        union = torch.sum(y_pred) + torch.sum(y_true)

    
        iou += (intersection + 1e-7) / (union - intersection + 1e-7) / B
    return iou

def remove_overlap(seg_out, warped_cm):
    
    assert len(warped_cm.shape) == 4
    
    warped_cm = warped_cm - (torch.cat([seg_out[:, 1:3, :, :], seg_out[:, 5:, :, :]], dim=1)).sum(dim=1, keepdim=True) * warped_cm
    return warped_cm

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="test")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="./data/")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--keep_step", type=int, default=300000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    # Cuda availability
    parser.add_argument('--cuda',default=False, help='cuda or cpu')
    # training
    parser.add_argument("--G_D_seperate", action='store_true')
    parser.add_argument("--no_GAN_loss", action='store_true')
    parser.add_argument("--lasttvonly", action='store_true')
    parser.add_argument("--interflowloss", action='store_true', help="Intermediate flow loss")
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
    parser.add_argument('--edgeawaretv', type=str, choices=['no_edge', 'last_only', 'weighted'], default="no_edge", help="Edge aware TV loss")
    parser.add_argument('--add_lasttv', action='store_true')
    
    # test visualize
    parser.add_argument("--no_test_visualize", action='store_true')    
    parser.add_argument("--num_test_visualize", type=int, default=3)
    parser.add_argument("--test_datasetting", default="unpaired")
    parser.add_argument("--test_dataroot", default="./data/")
    parser.add_argument("--test_data_list", default="test_pairs.txt")
    

    # Hyper-parameters
    parser.add_argument('--G_lr', type=float, default=0.0002, help='Generator initial learning rate for adam')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='Discriminator initial learning rate for adam')
    parser.add_argument('--CElamda', type=float, default=10, help='initial learning rate for adam')
    parser.add_argument('--GANlambda', type=float, default=1)
    parser.add_argument('--tvlambda', type=float, default=2)
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--val_count', type=int, default='1000')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")
    
    opt = parser.parse_args()
    return opt


def train(opt, train_loader, test_loader, val_loader, board, tocg, D):
    # Model
    tocg.cuda()
    tocg.train()
    D.cuda()
    D.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(opt)
    if opt.fp16:
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.HalfTensor)
    else :
        criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor)

    # optimizer
    optimizer_G = torch.optim.Adam(tocg.parameters(), lr=opt.G_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(0.5, 0.999))
    

    for step in tqdm(range(opt.load_step, opt.keep_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # input1
        c_paired = inputs['cloth']['paired'].cuda()
        cm_paired = inputs['cloth_mask']['paired'].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']

        # inputs
        input1 = torch.cat([c_paired, cm_paired], 1)
        input2 = torch.cat([parse_agnostic, densepose], 1)

        # forward
        flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
        
        # warped cloth mask one hot 
        
        warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        # fake segmap cloth channel * warped clothmask
        if opt.clothmask_composition != 'no_composition':
            if opt.clothmask_composition == 'detach':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                fake_segmap = fake_segmap * cloth_mask
                
            if opt.clothmask_composition == 'warp_grad':
                cloth_mask = torch.ones_like(fake_segmap.detach())
                cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                fake_segmap = fake_segmap * cloth_mask
        if opt.occlusion:
            warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
            warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)
        
        # generated fake cloth mask & misalign mask
        fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
        misalign = fake_clothmask - warped_cm_onehot
        misalign[misalign < 0.0] = 0.0
        
        # loss warping
        loss_l1_cloth = criterionL1(warped_clothmask_paired, parse_cloth_mask)
        loss_vgg = criterionVGG(warped_cloth_paired, im_c)

        loss_tv = 0
        
        if opt.edgeawaretv == 'no_edge':
            if not opt.lasttvonly:
                for flow in flow_list:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            else:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
        else:
            if opt.edgeawaretv == 'last_only':
                flow = flow_list[-1]
                warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                y_tv = y_tv * mask_y
                x_tv = x_tv * mask_x
                y_tv = y_tv.mean()
                x_tv = x_tv.mean()
                loss_tv = loss_tv + y_tv + x_tv
                
            elif opt.edgeawaretv == 'weighted':
                for i in range(5):
                    flow = flow_list[i]
                    warped_clothmask_paired_down = F.interpolate(warped_clothmask_paired, flow.shape[1:3], mode='bilinear')
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
                    mask_y = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, 1:, :, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :-1, :, :]))
                    mask_x = torch.exp(-150*torch.abs(warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, 1:, :] - warped_clothmask_paired_down.permute(0, 2, 3, 1)[:, :, :-1, :]))
                    y_tv = y_tv * mask_y
                    x_tv = x_tv * mask_x
                    y_tv = y_tv.mean() / (2 ** (4-i))
                    x_tv = x_tv.mean() / (2 ** (4-i))
                    loss_tv = loss_tv + y_tv + x_tv
            
            if opt.add_lasttv:
                for flow in flow_list[-1:]:
                    y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
                    x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
                    loss_tv = loss_tv + y_tv + x_tv
            

        N, _, iH, iW = c_paired.size()
        # Intermediate flow loss
        if opt.interflowloss:
            for i in range(len(flow_list)-1):
                flow = flow_list[i]
                N, fH, fW, _ = flow.size()
                grid = mkgrid(N, iH, iW)
                flow = F.interpolate(flow.permute(0, 3, 1, 2), size = c_paired.shape[2:], mode=opt.upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((fW - 1.0) / 2.0), flow[:, :, :, 1:2] / ((fH - 1.0) / 2.0)], 3)
                warped_c = F.grid_sample(c_paired, flow_norm + grid, padding_mode='border')
                warped_cm = F.grid_sample(cm_paired, flow_norm + grid, padding_mode='border')
                warped_cm = remove_overlap(F.softmax(fake_segmap, dim=1), warped_cm)
                loss_l1_cloth += criterionL1(warped_cm, parse_cloth_mask) / (2 ** (4-i))
                loss_vgg += criterionVGG(warped_c, im_c) / (2 ** (4-i))
            
        # loss segmentation
        # generator
        CE_loss = cross_entropy2d(fake_segmap, label_onehot.transpose(0, 1)[0].long())
        
        if opt.no_GAN_loss:
            loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda)
            # step
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        
        else:
            fake_segmap_softmax = torch.softmax(fake_segmap, 1)

            pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
            
            loss_G_GAN = criterionGAN(pred_segmap, True)
            
            if not opt.G_D_seperate:  
                # discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)

                # loss sum
                loss_G = (10 * loss_l1_cloth + loss_vgg +opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                loss_D = loss_D_fake + loss_D_real

                # step
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                
            else: # train G first after that train D
                # loss G sum
                loss_G = (10 * loss_l1_cloth + loss_vgg + opt.tvlambda * loss_tv) + (CE_loss * opt.CElamda + loss_G_GAN * opt.GANlambda)  # warping + seg_generation
                
                # step G
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                
                # discriminator
                with torch.no_grad():
                    _, fake_segmap, _, _ = tocg(input1, input2)
                fake_segmap_softmax = torch.softmax(fake_segmap, 1)
                
                # loss discriminator
                fake_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax.detach()),dim=1))
                real_segmap_pred = D(torch.cat((input1.detach(), input2.detach(), label),dim=1))
                loss_D_fake = criterionGAN(fake_segmap_pred, False)
                loss_D_real = criterionGAN(real_segmap_pred, True)
                
                loss_D = loss_D_fake + loss_D_real
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
        # Vaildation
        if (step + 1) % opt.val_count == 0:
            tocg.eval()
            iou_list = []
            with torch.no_grad():
                for cnt in range(2000//opt.batch_size):
                
                    inputs = val_loader.next_batch()
                    # input1
                    c_paired = inputs['cloth']['paired'].cuda()
                    cm_paired = inputs['cloth_mask']['paired'].cuda()
                    cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                    # input2
                    parse_agnostic = inputs['parse_agnostic'].cuda()
                    densepose = inputs['densepose'].cuda()
                    openpose = inputs['pose'].cuda()
                    # GT
                    label_onehot = inputs['parse_onehot'].cuda()  # CE
                    label = inputs['parse'].cuda()  # GAN loss
                    parse_cloth_mask = inputs['pcm'].cuda()  # L1
                    im_c = inputs['parse_cloth'].cuda()  # VGG
                    # visualization
                    im = inputs['image']
                    
                    input1 = torch.cat([c_paired, cm_paired], 1)
                    input2 = torch.cat([parse_agnostic, densepose], 1)
                    
                    # forward
                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
                
                    # fake segmap cloth channel * warped clothmask
                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask
                            
                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap.detach())
                            cloth_mask[:, 3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask
    
                    # calculate iou
                    iou = iou_metric(F.softmax(fake_segmap, dim=1).detach(), label)
                    iou_list.append(iou.item())

            tocg.train()
            board.add_scalar('val/iou', np.mean(iou_list), step + 1)
        
        # tensorboard
        if (step + 1) % opt.tensorboard_count == 0:
            # loss G
            board.add_scalar('Loss/G', loss_G.item(), step + 1)
            board.add_scalar('Loss/G/l1_cloth', loss_l1_cloth.item(), step + 1)
            board.add_scalar('Loss/G/vgg', loss_vgg.item(), step + 1)
            board.add_scalar('Loss/G/tv', loss_tv.item(), step + 1)
            board.add_scalar('Loss/G/CE', CE_loss.item(), step + 1)
            if not opt.no_GAN_loss:
                board.add_scalar('Loss/G/GAN', loss_G_GAN.item(), step + 1)
                # loss D
                board.add_scalar('Loss/D', loss_D.item(), step + 1)
                board.add_scalar('Loss/D/pred_real', loss_D_real.item(), step + 1)
                board.add_scalar('Loss/D/pred_fake', loss_D_fake.item(), step + 1)
            
            grid = make_grid([(c_paired[0].cpu() / 2 + 0.5), (cm_paired[0].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu()), ((densepose.cpu()[0]+1)/2),
                              (im_c[0].cpu() / 2 + 0.5), parse_cloth_mask[0].cpu().expand(3, -1, -1), (warped_cloth_paired[0].cpu().detach() / 2 + 0.5), (warped_cm_onehot[0].cpu().detach()).expand(3, -1, -1),
                              visualize_segmap(label.cpu()), visualize_segmap(fake_segmap.cpu()), (im[0]/2 +0.5), (misalign[0].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            board.add_images('train_images', grid.unsqueeze(0), step + 1)
            
            if not opt.no_test_visualize:
                inputs = test_loader.next_batch()
                # input1
                c_paired = inputs['cloth'][opt.test_datasetting].cuda()
                cm_paired = inputs['cloth_mask'][opt.test_datasetting].cuda()
                cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                # input2
                parse_agnostic = inputs['parse_agnostic'].cuda()
                densepose = inputs['densepose'].cuda()
                openpose = inputs['pose'].cuda()
                # GT
                label_onehot = inputs['parse_onehot'].cuda()  # CE
                label = inputs['parse'].cuda()  # GAN loss
                parse_cloth_mask = inputs['pcm'].cuda()  # L1
                im_c = inputs['parse_cloth'].cuda()  # VGG
                # visualization
                im = inputs['image']

                tocg.eval()
                with torch.no_grad():
                    # inputs
                    input1 = torch.cat([c_paired, cm_paired], 1)
                    input2 = torch.cat([parse_agnostic, densepose], 1)

                    # forward
                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
                    
                    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
                    if opt.clothmask_composition != 'no_composition':
                        if opt.clothmask_composition == 'detach':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_cm_onehot
                            fake_segmap = fake_segmap * cloth_mask
                            
                        if opt.clothmask_composition == 'warp_grad':
                            cloth_mask = torch.ones_like(fake_segmap)
                            cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                            fake_segmap = fake_segmap * cloth_mask
                    if opt.occlusion:
                        warped_clothmask_paired = remove_overlap(F.softmax(fake_segmap, dim=1), warped_clothmask_paired)
                        warped_cloth_paired = warped_cloth_paired * warped_clothmask_paired + torch.ones_like(warped_cloth_paired) * (1-warped_clothmask_paired)
                    
                    # generated fake cloth mask & misalign mask
                    fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
                    misalign = fake_clothmask - warped_cm_onehot
                    misalign[misalign < 0.0] = 0.0
                
                for i in range(opt.num_test_visualize):
                    grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                                    (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                                    visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                        nrow=4)
                    board.add_images(f'test_images/{i}', grid.unsqueeze(0), step + 1)
                tocg.train()
        
        # display
        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            if not opt.no_GAN_loss:
                print("step: %8d, time: %.3f\nloss G: %.4f, L1_cloth loss: %.4f, VGG loss: %.4f, TV loss: %.4f CE: %.4f, G GAN: %.4f\nloss D: %.4f, D real: %.4f, D fake: %.4f"
                    % (step + 1, t, loss_G.item(), loss_l1_cloth.item(), loss_vgg.item(), loss_tv.item(), CE_loss.item(), loss_G_GAN.item(), loss_D.item(), loss_D_real.item(), loss_D_fake.item()), flush=True)

        # save
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_step_%06d.pth' % (step + 1)),opt)
            save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_step_%06d.pth' % (step + 1)),opt)

def main():
    opt = get_opt()
    print(opt)
    print("Start to train %s!" % opt.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create train dataset & loader
    train_dataset = CPDataset(opt)
    train_loader = CPDataLoader(opt, train_dataset)
    
    # create test dataset & loader
    test_loader = None
    if not opt.no_test_visualize:
        train_bsize = opt.batch_size
        opt.batch_size = opt.num_test_visualize
        opt.dataroot = opt.test_dataroot
        opt.datamode = 'test'
        opt.data_list = opt.test_data_list
        test_dataset = CPDatasetTest(opt)
        opt.batch_size = train_bsize
        val_dataset = Subset(test_dataset, np.arange(2000))
        test_loader = CPDataLoader(opt, test_dataset)
        val_loader = CPDataLoader(opt, val_dataset)
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=4, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)
    
    # Load Checkpoint
    if not opt.tocg_checkpoint == '' and os.path.exists(opt.tocg_checkpoint):
        load_checkpoint(tocg, opt.tocg_checkpoint)

    # Train
    train(opt, train_loader, val_loader, test_loader, board, tocg, D)

    # Save Checkpoint
    save_checkpoint(tocg, os.path.join(opt.checkpoint_dir, opt.name, 'tocg_final.pth'),opt)
    save_checkpoint(D, os.path.join(opt.checkpoint_dir, opt.name, 'D_final.pth'),opt)
    print("Finished training %s!" % opt.name)


if __name__ == "__main__":
    main()