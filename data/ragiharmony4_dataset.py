import os.path
import torch
import random
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2
import random
import json

import torchvision.transforms as f
import time
class RAGIharmony4Dataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt, is_for_train):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.isTrain = is_for_train
        self._load_images_paths()
        self.transform = get_transform(opt)

        self.color_aug = f.ColorJitter(hue=[-0.5,0.5])
        # print(self.opt.addpos)
        self.dvt=opt.dvt
        self.dvt_transform = transforms.Compose([
            transforms.ToTensor(),transforms.Resize([448,448*1],antialias=True),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])

        self.resize = 256
        self.kshots=1
    def _load_images_paths(self,):
        if self.isTrain == True:
            print('loading training file...')

            self.trainreffile = os.path.join(self.opt.dataset_root, 'IHD_train_all.jsonl')
            # refs = []
            # self.ref_exists = []
            self.ref_paths,self.external_image_paths,self.external_mask_paths,self.external_gt_paths = [],[],[],[]
            with open(self.trainreffile, 'r') as reader:
                lines = reader.readlines()
                for line in lines:
                    items = json.loads(line)
                    name_parts = items['input'].split('_')
                    gt_path = items['input'].replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')

                    self.ref_paths.append(items['refs'])
                    self.external_image_paths.append(os.path.join(self.opt.dataset_root, items['input']))
                    self.external_mask_paths.append(os.path.join(self.opt.dataset_root, items['mask']))
                    self.external_gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))
        elif self.isTrain == False:
            print('loading test file...')
            self.trainreffile = os.path.join(self.opt.dataset_root, self.opt.jsonl_path+'.jsonl')
            self.ref_paths,self.external_image_paths,self.external_mask_paths,self.external_gt_paths = [],[],[],[]
            with open(self.trainreffile, 'r') as reader:
                lines = reader.readlines()
                for line in lines:
                    items = json.loads(line)
                    name_parts = items['input'].split('_')
                    gt_path = items['input'].replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')

                    self.ref_paths.append(items['refs'])
                    self.external_image_paths.append(os.path.join(self.opt.dataset_root, items['input']))
                    self.external_mask_paths.append(os.path.join(self.opt.dataset_root, items['mask']))
                    self.external_gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))
    def __getitem__(self, index):
        refs = []
        dinorefs = []
        if self.isTrain==True:
            externalindex = index
            if len(self.ref_paths[externalindex])>0:
                for i in range(self.kshots):
                    # trainser = i
                    refindex = random.choice([j for j in range(len(self.ref_paths[externalindex]))])
                    ref = Image.open(os.path.join(self.opt.dataset_root, self.ref_paths[externalindex][refindex])).convert('RGB')
                    ref = tf.resize(ref, [self.resize,self.resize])
                    refs.append(self.transform(ref))
                    dinorefs.append(self.dvt_transform(ref))
        else:
            externalindex = index
            for refindex in range(min(1,len(self.ref_paths[externalindex]))):
                ref = Image.open(os.path.join(self.opt.dataset_root, self.ref_paths[externalindex][refindex])).convert('RGB')
                ref = tf.resize(ref, [self.resize,self.resize])
                refs.append(self.transform(ref))
                dinorefs.append(self.dvt_transform(ref))

        externalmask = Image.open(self.external_mask_paths[externalindex]).convert('1')
        externalcomp = Image.open(self.external_image_paths[externalindex]).convert('RGB')
        externalreal = Image.open(os.path.join(self.opt.dataset_root, self.external_gt_paths[externalindex])).convert('RGB')

        externalreal = tf.resize(externalreal, [self.resize,self.resize])
        externalmask = tf.resize(externalmask, [self.resize,self.resize])
        externalcomp = tf.resize(externalcomp, [self.resize,self.resize])

        if self.isTrain==True:
            aug_pos_ref,aug_pos_ref_dvt = self.aug_c(externalreal,externalmask)

            refs.append(aug_pos_ref)
            dinorefs.append(aug_pos_ref_dvt)
            c = list(zip(refs,refs))
            np.random.shuffle(c)
            augindex = random.choice([i for i in range(len(refs))])
            refs = refs[augindex]
            dinorefs = dinorefs[augindex]
        else:
            augindex = random.choice([i for i in range(len(refs))])
            refs = refs[augindex]
            dinorefs = dinorefs[augindex]



        exter_comp = self.transform(externalcomp)
        exter_comp_dvt = self.dvt_transform(externalcomp)


        externalreal = self.transform(externalreal)
        externalmask = tf.to_tensor(externalmask)
        exter_comp = self._compose(exter_comp, externalmask, externalreal)

        if self.isTrain==True:

            return {'comp_dvt':exter_comp_dvt,'extet_comp_dvt':exter_comp_dvt,'ref_dvt':dinorefs,'comp': exter_comp, 'mask': externalmask, 'real': externalreal , 'ref':refs,'extet_comp':exter_comp,'externalmask':externalmask,'externalreal':externalreal,'img_path':self.external_image_paths[index],'ref_path':self.external_image_paths[index]}
        else:
            return {'comp_dvt':exter_comp_dvt,'extet_comp_dvt':exter_comp_dvt,'ref_dvt':dinorefs,'comp': exter_comp, 'mask': externalmask, 'real': externalreal , 'ref':refs,'extet_comp':exter_comp,'externalmask':externalmask,'externalreal':externalreal,'img_path':self.external_image_paths[index],'ref_path':self.ref_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        if self.isTrain==True:

            return len(self.external_image_paths)
        else:
            return len(self.external_image_paths)
    def _compose(self, foreground_img, foreground_mask, background_img):
        return foreground_img * foreground_mask + background_img * (1 - foreground_mask)
    # def generate_mask(self,)
    def aug_c(self,input,mask):
        centersize = 196
        mask0 = np.array(mask).astype(np.uint8)
        mask0 = mask0[:,:,None]
        if_ok = False
        for nums in range(5):

            temp_h_s = random.randint(0,self.resize-centersize)
            temp_w_s = random.randint(0,self.resize-centersize)
            mask_temp = np.zeros([self.resize,self.resize,1])
            mask_temp[temp_h_s:temp_h_s+centersize,temp_w_s:temp_w_s+centersize,:]=1
            if np.sum(mask_temp*mask0) > (np.sum(mask0))*0.5:
                eh = temp_h_s + centersize
                ew = temp_w_s + centersize
                if_ok = True
                break
        if if_ok==False:
            temp_h_s=0
            temp_w_s =0
            eh=256
            ew = 256            
        pos_ref = np.array(input)[temp_h_s:eh,temp_w_s:ew,:]
        pos_ref = np.ascontiguousarray(pos_ref)
        pos_ref = cv2.resize(pos_ref,[self.resize,self.resize])
        if random.randint(0,1)==1:
            pos_ref = pos_ref[:,::-1,:]
        aug_output_temp = Image.fromarray(pos_ref)
        aug_output = self.transform(aug_output_temp)
        aug_output_dvt = self.dvt_transform(aug_output_temp)


        return aug_output,aug_output_dvt