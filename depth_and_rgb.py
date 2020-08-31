import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
path = './TUM_RGBD_dataset/rgbd_dataset_freiburg1_room/depth/1305031910.771502.png'

img = cv2.imread(path)

t1 = img[:, :, 0]

t = np.repeat(t1[:, :, np.newaxis], 3, axis=2)
print(t.shape)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img[:, :, 0])
ax[1].imshow(img[:, :, 1])
ax[2].imshow(img[:, :, 2])
plt.show()

exit()
"""


import argparse
from video_to_images import save_frame_range_sec
import sys
import tensorflow as tf
sys.path.append('DepthNet')
from depthnet_model import *
from manage_depth_image import get_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='vgg')
parser.add_argument('--video_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--image_height', type=int, default=256)
parser.add_argument('--image_width', type=int, default=512)
        
args = parser.parse_args()

rgb_dirs = 'test_dataset/rgb/'
depth_dirs = 'test_dataset/depth/'

if not os.path.exists(rgb_dirs):
    os.mkdir(rgb_dirs)

if not os.path.exists(depth_dirs):
    os.mkdir(depth_dirs)

rgb_image_pathes = os.listdir(rgb_dirs)
if len(rgb_image_pathes) == 0:
    # create rgb dataset with timestamp name
    cap = cv2.VideoCapture(args.video_path)
    # get the number of frame
    video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_len_sec = video_frame / video_fps

    print('total video length:', video_len_sec)
    save_frame_range_sec(args.video_path, 0, video_len_sec, 0.5, rgb_dirs, '2006031220')


# build model
params = depthnet_parameters(
    encoder=args.encoder,
    height=args.image_height,
    width=args.image_width,
    batch_size=2,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False
)

left = tf.placeholder(tf.float32, [2, args.image_height, args.image_width, 3])
mode = 'test'
model = DepthNet(params, mode, left, None)

model_set = (model, left)


rgb_image_pathes = sorted([p[:-4] for p in rgb_image_pathes])
rgb_image_pathes = [rgb_dirs + '/' + rp + '.png' for rp in rgb_image_pathes]

for rgb_path in rgb_image_pathes:
    image = cv2.imread(rgb_path)
    print('rgb shape:', image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image.shape[:2]
    
    depth = get_depth_image(model_set, image, args)
    print('depth shape:', depth.shape)
    
    filename = depth_dirs + rgb_path.split('/')[-1]
    
    # save depth image
    fig = plt.figure()
    plt.imshow(depth)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    
    depth_image = cv2.imread(filename, 0)
    depth_image = cv2.resize(depth_image, (width, height))
    cv2.imwrite(filename, depth_image)
    
    
    print('filename:', filename)




