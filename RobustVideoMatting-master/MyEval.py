import torch
from model import MattingNetwork
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2


TRIMAP_CHANNEL = 1

RANDOM_INTERP = True

CUTMASK_PROB = 0

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
               cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

crop_size=768
def maybe_random_interp(cv2_interp):
    if RANDOM_INTERP:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class CropResize(object):
    # crop the image to square, and resize to target size
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # crop the image to square, and resize to target size
        img,alpha=sample['image'],sample['alpha']
        h, w = img.shape[:2]
        if h == w:
            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        elif h > w:
            margin = (h-w)//2
            img = img[margin:margin+w, :]
            alpha = alpha[margin:margin+w, :]

            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        else:
            margin = (w-h)//2
            img = img[:, margin:margin+h]
            alpha = alpha[:, margin:margin+h]

            img_crop = cv2.resize(
                img, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(
                alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        sample.update({'image': img_crop, 'alpha': alpha_crop})
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __init__(self, phase="test", norm_type='sd'):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase
        self.norm_type = norm_type

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha = sample['image'][:, :, ::-
        1], sample['alpha']

        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)




        sample['image'], sample['alpha'] = \
            torch.from_numpy(image), torch.from_numpy(
                alpha)
        sample['image'] /= 255.

        return sample

def load_image(image_,alpha_):
    image = cv2.imread(image_)
    alpha = cv2.imread(alpha_, 0) / 255.
    sample = {}
    sample['image'] = image
    sample['alpha'] = alpha
    sample = transform(sample)
    return sample['image'].unsqueeze(0), sample['alpha'].unsqueeze(0)

if __name__=='__main__':

    model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
    model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
    test_file = "../MyData/image"
    test_file_alpha = "../MyData/alpha"
    test_image_file_list = []
    test_alpha_file_list = []
    transform = transforms.Compose([CropResize(
        (crop_size, crop_size)), ToTensor(phase="val", norm_type="sd")])
    for i in range(1, 121):
        test_image_file_list.append(os.path.join(test_file, f'{i}.png'))
        test_alpha_file_list.append(os.path.join(test_file_alpha, f'{i}.png'))

 # Green background.
    rec = [None] * 4                                       # Initial recurrent states.
    output_dir = 'output_frames'
    true_alpha_dir='true_alpha'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(true_alpha_dir, exist_ok=True)
    downsample_ratio=0.375
    with torch.no_grad():
        for i in range(1,121): # RGB tensor normalized to 0 ~ 1.
            src,true_alpha=load_image(test_image_file_list[i-1],test_alpha_file_list[i-1])
            fgr, pha, *rec = model(src.cuda(), *rec, downsample_ratio)
            pha_image = (pha[0].cpu().numpy() * 255).astype(np.uint8)  # Assume batch size of 1
            pha_image = np.transpose(pha_image, (1, 2, 0))  # Change from CxHxW to HxWxC
            pha_image = Image.fromarray(pha_image.squeeze(), mode='L')  # Squeeze if single channel
            true_pha_image = (true_alpha[0].cpu().numpy() * 255).astype(np.uint8)  # Assume batch size of 1
            true_pha_image = np.transpose(true_pha_image, (1, 2, 0))  # Change from CxHxW to HxWxC
            true_pha_image = Image.fromarray(true_pha_image.squeeze(), mode='L')  # Squeeze if single channel
            # Save the image
            pha_image.save(os.path.join(output_dir, f'{i}.png'))
            true_pha_image.save(os.path.join(true_alpha_dir, f'{i}.png'))
