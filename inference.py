import os
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

input_path='inputs'
output_path='outputs'
ckpt_path='modnet_photographic_portrait_matting.ckpt'
# check input arguments
if not os.path.exists(input_path):
    print('Cannot find input path: {0}'.format(input_path))
    exit()
if not os.path.exists(output_path):
    print('Cannot find output path: {0}'.format(output_path))
    exit()
if not os.path.exists(ckpt_path):
    print('Cannot find ckpt path: {0}'.format(ckpt_path))
    exit()

# define hyper-parameters
ref_size = 512

# define image to tensor transform
im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# create MODNet and load the pre-trained ckpt
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load(ckpt_path))
modnet.eval()


def save_fg(img_data,matte_data,filename):
    w,h = img_data.width,img_data.height
    img_data = np.asarray(img_data)
    if len(img_data.shape) ==2:
       img_data = img_data[:,:,None]
    if img_data.shape[2]==1:
       img_data = np.repeat(img_data,3,axis=2)
    elif img_data.shape[2] ==4:
       img_data = img_data[:,:,0:3]
    matte = np.repeat(matte_data[:,:,None],3,axis=2)
    # fg_black = matte*img_data#+(1-matte)*np.full(img_data.shape,0)
    fg_white = matte*img_data+(1-matte)*np.full(img_data.shape,255)
    # combined = np.concatenate((img_data,matte*255,fg_black,fg_white),axis=1)
    output = Image.fromarray(np.uint8(fg_white))
    output.save(filename)
    return


# inference images
im_names = os.listdir(input_path)
for im_name in im_names:
    print('Process image: {0}'.format(im_name))

    # read image
    im = Image.open(os.path.join(input_path, im_name))
    ori=im.copy()
    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda(), True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    print(matte)
    matte_name = im_name.split('.')[0] + '_mask.png'
    res_name=im_name.split('.')[0] + '_result.png'
    # print(matte.shape)
    # fg = matte * ori + (1 - matte) * np.full(ori.shape, 0.0)
    Image.fromarray(((matte*255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))
    save_fg(ori,matte,os.path.join(output_path,res_name))
    # print(matte[0])
    # matte= matte.convert("RGB")
    # print(matte.size)
    # fg = matte * ori + (1-matte)*np.full(ori.shape, 0.0)
    # print(matte.size)
    # matte= cv2.cvtColor(np.asarray(matte), cv2.COLOR_RGB2BGR)
    # img = cv2.imread(os.path.join(input_path, im_name))
    # matting = cv2.imread(matte, cv2.IMREAD_GRAYSCALE)
    # fg = matte * img + (1 - matte) * np.full(img.shape, 0.0)
    # ori = Image.fromarray(((matte * 255).astype('uint8')))
    # ori=Image.fromarray(ori)
    # fg = matte * ori + (1 - matte) * np.full(ori.shape, 0.0)
    # fg.save(os.path.join(output_path, matte_name))
    # mask.save('mask.png')
    # ori = cv2.cvtColor(np.asarray(ori), cv2.COLOR_RGB2BGR)
    # mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)
    # mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('mask2.png',mask)
    # ori[mask == 0] = (255, 255, 255)
    # cv2.imwrite(os.path.join(output_path, matte_name),ori)


