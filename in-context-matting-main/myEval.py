import argparse
from omegaconf import OmegaConf
from icm.util import instantiate_from_config
import torch
from pytorch_lightning import Trainer, seed_everything
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image


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

        if self.norm_type == 'imagenet':
            # normalize image
            sample['image'] /= 255.

            sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        elif self.norm_type == 'sd':
            sample['image'] = sample['image'].to(dtype=torch.float32) / 127.5 - 1.0
        else:
            raise NotImplementedError(
                "norm_type {} is not implemented".format(self.norm_type))

        return sample
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="12-0.00800-mat.pth",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args

def load_image(image_,alpha_):
    image = cv2.imread(image_)
    alpha = cv2.imread(alpha_, 0) / 255.
    sample = {}
    sample['image'] = image
    sample['alpha'] = alpha
    sample = transform(sample)
    return sample['image'], sample['alpha']



if __name__ == '__main__':
    args = parse_args()
    output_dir = 'output_frames'
    true_alpha_dir = 'true_alpha'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(true_alpha_dir, exist_ok=True)
    test_file = "../MyData/image"
    test_file_alpha = "../MyData/alpha"
    test_image_file_list=[]
    test_alpha_file_list=[]
    transform=transforms.Compose([CropResize(
        (crop_size, crop_size)), ToTensor(phase="val",norm_type="sd")])
    for i in range(1,121):
        test_image_file_list.append(os.path.join(test_file,f'{i}.png'))
        test_alpha_file_list.append(os.path.join(test_file_alpha,f'{i}.png'))

    reference_image,reference_alpha=load_image(test_image_file_list[0],test_alpha_file_list[0])
    seed_everything(args.seed)
    cfg = OmegaConf.load(args.config)
    """=== Init model ==="""
    cfg_model = cfg.get('model')
    # model = instantiate_from_config(cfg_model)
    model = load_model_from_config(cfg_model, args.checkpoint, verbose=True)
    model.cuda()
    model.eval()
    for i in range(1,120):
        test_image,true_alpha=load_image(test_image_file_list[i],test_alpha_file_list[i])
        output, cross_map, self_map=model(reference_image.unsqueeze(0).to('cuda'),reference_alpha.unsqueeze(0).to('cuda'),test_image.unsqueeze(0).to('cuda'))
        pha_image = (output[0].cpu().detach().numpy() * 255).astype(np.uint8)  # Assume batch size of 1
        pha_image = np.transpose(pha_image, (1, 2, 0))  # Change from CxHxW to HxWxC
        pha_image = Image.fromarray(pha_image.squeeze(), mode='L')  # Squeeze if single channel
        true_pha_image = (true_alpha.numpy() * 255).astype(np.uint8)  # Assume batch size of 1
        true_pha_image = np.transpose(true_pha_image, (1, 2, 0))  # Change from CxHxW to HxWxC
        true_pha_image = Image.fromarray(true_pha_image.squeeze(), mode='L')  # Squeeze if single channel
        # Save the image
        pha_image.save(os.path.join(output_dir, f'{i+1}.png'))
        true_pha_image.save(os.path.join(true_alpha_dir, f'{i+1}.png'))

    # i=79
    # reference_image, reference_alpha = load_image(test_image_file_list[i-1], test_alpha_file_list[i-1])
    # test_image, true_alpha = load_image(test_image_file_list[i], test_alpha_file_list[i])
    # output, cross_map, self_map = model(reference_image.unsqueeze(0).to('cuda'),
    #                                     reference_alpha.unsqueeze(0).to('cuda'), test_image.unsqueeze(0).to('cuda'))
    # pha_image = (output[0].cpu().detach().numpy() * 255).astype(np.uint8)  # Assume batch size of 1
    # pha_image = np.transpose(pha_image, (1, 2, 0))  # Change from CxHxW to HxWxC
    # pha_image = Image.fromarray(pha_image.squeeze(), mode='L')  # Squeeze if single channel
    #
    # # Save the image
    # pha_image.save(f'{i+1}.png')


