import torch
import torch.nn as nn

from torchvision.utils import make_grid, save_image

import argparse
import os
import time
from cp_dataset import CPDatasetTest, CPDataLoader
from networks import ConditionGenerator, load_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from get_norm_const import D_logit


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default="./data/zalando-hd-resize")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--datasetting", default="paired")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='', help='tocg checkpoint')
    parser.add_argument('--D_checkpoint', type=str, default='', help='D checkpoint')
    
    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        

    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")
    
    # Discriminator
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--norm_const', type=float, help='Normalizing constant for rejection sampling')
    
    opt = parser.parse_args()
    return opt


def test(opt, test_loader, board, tocg, D=None):
    # Model
    tocg.cuda()
    tocg.eval()
    if D is not None:
        D.cuda()
        D.eval()
    
    os.makedirs(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'multi-task'), exist_ok=True)
    num = 0
    iter_start_time = time.time()
    if D is not None:
        D_score = []
    for inputs in test_loader.data_loader:
        
        # input1
        c_paired = inputs['cloth'][opt.datasetting].cuda()
        cm_paired = inputs['cloth_mask'][opt.datasetting].cuda()
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

        with torch.no_grad():
            # inputs
            input1 = torch.cat([c_paired, cm_paired], 1)
            input2 = torch.cat([parse_agnostic, densepose], 1)

            # forward
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(input1, input2)
            
            # warped cloth mask one hot 
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
            if D is not None:
                fake_segmap_softmax = F.softmax(fake_segmap, dim=1)
                pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
                score = D_logit(pred_segmap)
                # score = torch.exp(score) / opt.norm_const
                score = (score / (1 - score)) / opt.norm_const
                print("prob0", score)
                for i in range(cm_paired.shape[0]):
                    name = inputs['c_name']['paired'][i].replace('.jpg', '.png')
                    D_score.append((name, score[i].item()))
            
            
            # generated fake cloth mask & misalign mask
            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_cm_onehot
            misalign[misalign < 0.0] = 0.0
        
        for i in range(c_paired.shape[0]):
            grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                            (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1), (warped_cloth_paired[i].cpu().detach() / 2 + 0.5), (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                            visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            save_image(grid, os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'multi-task',
                             (inputs['c_name']['paired'][i].split('.')[0] + '_' +
                              inputs['c_name']['unpaired'][i].split('.')[0] + '.png')))
        num += c_paired.shape[0]
        print(num)
    if D is not None:
        D_score.sort(key=lambda x: x[1], reverse=True)
        # Save D_score
        for name, score in D_score:
            f = open(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                                opt.datamode, opt.datasetting, 'multi-task', 'rejection_prob.txt'), 'a')
            f.write(name + ' ' + str(score) + '\n')
            f.close()
    print(f"Test time {time.time() - iter_start_time}")


def main():
    opt = get_opt()
    print(opt)
    print("Start to test %s!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1], opt.datamode, opt.datasetting))

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        if opt.norm_const is None:
            raise NotImplementedError
        D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)
    else:
        D = None
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        load_checkpoint(D, opt.D_checkpoint)
    # Train
    test(opt, test_loader, board, tocg, D=D)

    print("Finished testing!")


if __name__ == "__main__":
    main()