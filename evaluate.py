import argparse
import os

import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.models.inception import inception_v3

import eval_models as models


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='LPIPS')
    parser.add_argument('--predict_dir', default='./result/bg_ver1/output/')
    parser.add_argument('--ground_truth_dir', default='./data/zalando-hd-resize/test/image')
    parser.add_argument('--resolution', type=int, default=1024)
    

    opt = parser.parse_args()
    return opt

def Evaluation(opt, pred_list, gt_list):
    T1 = Transforms.ToTensor()
    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])
    T3 = Transforms.Compose([Transforms.Resize((299, 299)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])

    splits = 1 # Hyper-parameter for IS score

    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)
    model.eval()
    inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
    inception_model.eval()

    avg_ssim, avg_mse, avg_distance = 0.0, 0.0, 0.0
    preds = np.zeros((len(gt_list), 1000))
    lpips_list = []
    with torch.no_grad():
        print("Calculate SSIM, MSE, LPIPS...")
        for i, img_pred in enumerate(pred_list):
            img = img_pred.split('_')[0] + '_00.jpg'
            # Calculate SSIM
            gt_img = Image.open(os.path.join(opt.ground_truth_dir, img))
            if not opt.resolution == 1024:
                if opt.resolution == 512:
                    gt_img = gt_img.resize((384,512), Image.BILINEAR)
                elif opt.resolution == 256:
                    gt_img = gt_img.resize((192,256), Image.BILINEAR)
                else:
                    raise NotImplementedError
            
            gt_np = np.asarray(gt_img.convert('L'))
            pred_img = Image.open(os.path.join(opt.predict_dir, img_pred))
            assert gt_img.size == pred_img.size, f"{gt_img.size} vs {pred_img.size}"
            pred_np = np.asarray(pred_img.convert('L'))
            avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)

            # Calculate LPIPS
            gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()
            pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            lpips_list.append((img_pred, model.forward(gt_img_LPIPS, pred_img_LPIPS).item()))
            avg_distance += lpips_list[-1][1]
            # Calculate Inception model prediction
            pred_img_IS = T3(pred_img).unsqueeze(0).cuda()
            preds[i] = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy()

            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)

            print(f"step: {i+1} evaluation... lpips:{lpips_list[-1][1]}")

        avg_ssim /= len(gt_list)
        avg_mse = avg_mse / len(gt_list)
        avg_distance = avg_distance / len(gt_list)

        # Calculate Inception Score
        split_scores = [] # Now compute the mean kl-divergence

        lpips_list.sort(key=lambda x: x[1], reverse=True)
        for name, score in lpips_list:
            f = open(os.path.join(opt.predict_dir, 'lpips.txt'), 'a')
            f.write(f"{name} {score}\n")
            f.close()
        print("Calculate Inception Score...")
        for k in range(splits):
            part = preds[k * (len(gt_list) // splits): (k+1) * (len(gt_list) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        IS_mean, IS_std = np.mean(split_scores), np.std(split_scores)
    f = open(os.path.join(opt.predict_dir, 'eval.txt'), 'a')
    f.write(f"SSIM : {avg_ssim} / MSE : {avg_mse} / LPIPS : {avg_distance}\n")
    f.write(f"IS_mean : {IS_mean} / IS_std : {IS_std}\n")
    
    f.close()
    return avg_ssim, avg_mse, avg_distance, IS_mean, IS_std



def main():
    opt = get_opt()

    # Output과 Ground Truth Data
    pred_list = os.listdir(opt.predict_dir)
    gt_list = os.listdir(opt.ground_truth_dir)
    pred_list.sort()
    gt_list.sort()

    avg_ssim, avg_mse, avg_distance, IS_mean, IS_std = Evaluation(opt, pred_list, gt_list)
    print("SSIM : %f / MSE : %f / LPIPS : %f" % (avg_ssim, avg_mse, avg_distance))
    print("IS_mean : %f / IS_std : %f" % (IS_mean, IS_std))


if __name__ == '__main__':
    main()
