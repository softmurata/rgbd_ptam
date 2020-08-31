import cv2
import os

def save_frame_range_sec(video_path, start_sec, stop_sec, step_sec, dir_path,
basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    print(digit)
    exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_inv = 1.0 / fps

    sec = start_sec

    while sec < stop_sec:
        print('sec:', sec)
        n = round(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        # read frame
        ret, frame = cap.read()

        if ret:
            target = round(n * fps_inv, 2)
            ss = str(n).zfill(digit)
            cv2.imwrite('{}.{}.png'.format(base_path, ss),frame)

        else:
            return

        sec += step_sec









if __name__ == '__main__':
    cap = cv2.VideoCapture('./videos/human0.mp4')
    # get the number of frame
    video_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_len_sec = video_frame / video_fps

    print('total video length:', video_len_sec)

    save_frame_range_sec('./videos/human0.mp4', 0, video_len_sec, 0.5, './videos/results', '2006031220')
