from torch import nn
import random
import torch
import torchvision.transforms.functional as F


class AugTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,imgs,masks):
        '''
        params:
         imgs: (b,3,h,w)
         masks: (b,h,w)
        '''
        img_aug = [imgs]
        mask_aug = [masks]
        if random.random() > 0.5:
            img_aug.append(F.hflip(imgs))
            mask_aug.append(F.hflip(masks))
        else:
            angle = random.choice([-30, -15, 0, 15, 30])
            img_aug.append(F.rotate(imgs, angle))
            mask_aug.append(F.rotate(masks,angle))
        return torch.cat(img_aug,dim=0),torch.cat(mask_aug,dim=0)