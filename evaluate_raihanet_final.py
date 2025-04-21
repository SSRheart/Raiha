import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import CustomDataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage import data, io
import cv2
import jsonlines
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def calculateMean(vars):
    return sum(vars) / len(vars)
# def calculateMean(vars,mask):
    # print(vars)
    # return np.sum(np.array(vars)*mask) / np.sum(mask)#len(vars)

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)

def evaluateModel(epoch_number, model, opt, test_dataset, epoch, max_psnr, iters=None):
    
    model.netG.eval()
    save_path = os.path.join('evaluate', opt.name, str(opt.epoch),opt.jsonl_path+'.jsonl')
    save = opt.saveimg


    os.makedirs(os.path.join('evaluate', opt.name, str(opt.epoch)),exist_ok=True)

    attslists = []
    test_idx = 0
    items = []
    metrics = {}
    tempname = None

    with jsonlines.open(save_path,'w' ) as writer:

        for i, data in tqdm(enumerate(test_dataset)):
            # print(i)
            model.set_input(data)  # unpack data from data loader
            model.test()  # inference
            visuals = model.get_current_visuals()  # get image results
            output = visuals['attentioned']
            real = visuals['real']
            reoutput = visuals['reattentioned']
            atts = visuals['attmap']
            refs = visuals['ref']
            inputs = visuals['comp']
            masks = visuals['mask']
            metrics = np.zeros([400,4,4])
            oldpaths = []
            for i_img in range(real.size(0)):
                img_path = data['img_path'][i_img]
                ref_path = data['ref_path'][i_img][0][0:]
                oldpaths.append(img_path)

                gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
                predref = reoutput[i_img:i_img+1]
                atts_  = atts[i_img:i_img+1]
                ref = refs[i_img:i_img+1]
                nowindex = '/'.join(img_path.split('/')[-3:])
                att_rgb = util.tensor2att(atts_)

                mask = masks[i_img:i_img+1]
                inputs_ = inputs[i_img:i_img+1]
                testname = '/'.join(img_path.split('/')[-3:])
                area = mask.sum().item()
                ref_path = ref_path.split('/')[-1]
                fore_nums = data['mask'][i_img].sum().item()
                mse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt))
                psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
                ssim_score = ssim(util.tensor2im(pred), util.tensor2im(gt),data_range=255)
                ref_mse_score_op = mean_squared_error(util.tensor2im(predref), util.tensor2im(gt))
                ref_psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(predref), data_range=255)
                ref_ssim_score = ssim(util.tensor2im(predref), util.tensor2im(gt),data_range=255)
                items = {
                            'imgpath': img_path,
                            'refpath':ref_path,
                            'PSNR': str(psnr_score_op),
                            'MSE': str(mse_score_op),
                            'ref_PSNR': str(ref_psnr_score_op),
                            'ref_MSE': str(ref_mse_score_op),
                            'area': str(area),
                        }
                writer.write(items)
                if save == 1.0:
                    mask_rgb = util.tensor2im(mask)
                    pred_rgb = util.tensor2im(pred)
                    ref_rgb = util.tensor2im(ref)
                    inputs_rgb = util.tensor2im(inputs_)
                    gt_rgb = util.tensor2im(gt)
                    predref_rgb = util.tensor2im(predref)

                    refscore = 1
                    basename, imagename = os.path.split(img_path)

                    basename = basename.split('/')[-2]
                    contours,_ = cv2.findContours((mask_rgb[:,:,0:1]/255.0).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    inputs_rgb = np.array(inputs_rgb)
                    inputs_rgb = cv2.resize(inputs_rgb,[256,256])
                    inputs_rgb_ = inputs_rgb

                    normed_mask = np.uint8(att_rgb)#[:,0:256:,:]
                    normed_mask = cv2.applyColorMap(normed_mask,cv2.COLORMAP_JET)
                    normed_mask = cv2.addWeighted(np.concatenate((inputs_rgb,ref_rgb),1),0.6,normed_mask,1.0,0)[:,:,::-1]
                    img111 = cv2.drawContours(inputs_rgb,contours,-1,(0,0,255),3)
                    save_internal = pred_rgb
                    save_external = predref_rgb
                    save_att = normed_mask
                    save_ref = ref_rgb
                    t,binary = cv2.threshold(mask_rgb,127,255,cv2.THRESH_BINARY)

                    contours,hierarchy=cv2.findContours(image=binary[:,:,0].astype(np.uint8),mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(inputs_rgb,contours,-1,(255,0,0),2)

                    save_img(os.path.join('evaluate', opt.name, str(opt.epoch),'res',opt.jsonl_path,basename, imagename.split('.')[0] +'_internal.png'), save_internal)
                    save_img(os.path.join('evaluate', opt.name, str(opt.epoch),'res',opt.jsonl_path,basename, imagename.split('.')[0] +'.png'), save_external)
                    save_img(os.path.join('evaluate', opt.name, str(opt.epoch),'res',opt.jsonl_path,basename, imagename.split('.')[0] +'_ref.png'), save_ref)
                    save_img(os.path.join('evaluate', opt.name, str(opt.epoch),'res',opt.jsonl_path,basename, imagename.split('.')[0] +'_comp.png'), inputs_rgb)
                    save_img(os.path.join('evaluate', opt.name, str(opt.epoch),'res',opt.jsonl_path,basename, imagename.split('.')[0] +'_gt.png'), gt_rgb)



    with jsonlines.open(save_path, 'r') as reader:
        records = list(reader)[0:]
    # print(len(records))
    preds = []
    gts = []  # 
    results = {}
    psnrs = []
    nums = 0
    for r in records:
        # out_psnr = []
        imgpath = r['imgpath']
        # score = float(r['ref_score'])
        # if score <=threshold:

        out_psnr = float(r['PSNR'])
        out_mse = float(r['MSE'])

        out_refpsnr = float(r['ref_PSNR'])
        out_refmse = float(r['ref_MSE'])
        results[str(nums)] = [[out_mse,out_psnr,out_refmse,out_refpsnr]]
        nums = nums + 1
    # print(len(results.keys()))
    oursmetrics = []
    for r in results:
        metrics_temp = np.mean(np.array(results[r]),0)
        oursmetrics.append(metrics_temp)
    print(len(oursmetrics),np.mean(np.array(oursmetrics),0))
def updateWriterInterval(writer, metrics, epoch):
    for k, v in metrics.items():
        writer.add_scalar('interval/{}-MSE'.format(k), v[0], epoch)
        writer.add_scalar('interval/{}-PSNR'.format(k), v[1], epoch)

if __name__ == '__main__':
    # setup_seed(6)
    opt = TestOptions().parse()   # get training 
    # train_dataset = CustomDataset(opt, is_for_train=True)
    test_dataset = CustomDataset(opt, is_for_train=False)
    # train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    test_dataset_size = len(test_dataset)
    # print('The number of training images = %d' % train_dataset_size)
    print('The number of testing images = %d' % test_dataset_size)
    
    # train_dataloader = train_dataset.load_data()
    test_dataloader = test_dataset.load_data()
    # print('The total batches of training images = %d' % len(train_dataset.dataloader))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name))

    max_psnr = 0
    max_epoch = 1
    for epoch in range(1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = opt.epoch                # the number of training iterations in current epoch, reset to 0 every epoch


        evaluateModel(epoch, model, opt, test_dataloader, epoch, max_psnr)

