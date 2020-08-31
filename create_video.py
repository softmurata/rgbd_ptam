import cv2
import os
import matplotlib.pyplot as plt

frame_size = -1
fps = 30

image_dir_path = './TUM_RGBD_dataset/rgbd_dataset_freiburg1_xyz/rgb'


img_datas = os.listdir(image_dir_path)
img_datas = sorted([img[:-4] for img in img_datas])
img_datas = [image_dir_path + '/' + img + '.png' for img in img_datas]


img_width, img_height, _ = cv2.imread(img_datas[0]).shape

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('test.mp4', fourcc, fps, (img_width, img_height))

print('formatting videos')
for img_path in img_datas:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_width, img_height))
    video.write(img)
    
    
video.release()
print('finish creating videos')
        
        



