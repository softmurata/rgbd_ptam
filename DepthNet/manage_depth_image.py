import numpy as np
import cv2
import tensorflow as tf

def post_process_for_disparity(disp_image):
    
    _, h, w = disp_image.shape
    l_disp = disp_image[0, :, :]
    r_disp = np.fliplr(disp_image[1, :, :])
    # mean disparity
    m_disp = 0.5 * (l_disp + r_disp)
    # meshgrid
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    
    return l_mask * r_disp + r_mask * l_disp + (1.0 - l_mask - r_mask) * m_disp

def get_depth_image(depthnet_model_set, rgb_image, args):
    
    model, left = depthnet_model_set
    
    # molding input images
    original_height, original_width, num_channels = rgb_image.shape
    # resize image for adjusting NN
    input_image = cv2.resize(rgb_image, (args.image_width, args.image_height), interpolation=cv2.INTER_LANCZOS4)
    input_image = input_image.astype(np.float32) / 255  # normalization
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    
    # tensorflow session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    
    # saver
    train_saver = tf.train.Saver()
    
    # Initialize tensorflow variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    
    # restore(need checkpoint path)
    restore_path = args.checkpoint_path.split('.')[0]
    train_saver.restore(sess, restore_path)
    
    disp_image = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    
    # post process
    disp_image = disp_image.squeeze()
    disp_pp = post_process_for_disparity(disp_image)
    
    disp_pp = disp_pp.astype(np.float32)
    disp_pp = disp_pp.squeeze()
    disp_image = cv2.resize(disp_pp, (original_width, original_height))  # (height, width, 1) => if you want to (height, width), disp_pp.squeeze()
    
    return disp_image
