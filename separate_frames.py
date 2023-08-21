import os
import cv2
import glob

def capture_frames(video_path, save_dir, interval=1):
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_no = 0
    i = 0
    while video.isOpened():
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = video.read()
        if not ret:
            break

        cv2.imwrite(os.path.join(save_dir, f'frame_{i}.jpg'), frame)
        i += 1

        frame_no += int(fps * interval)

    video.release()
    print(f'Finished saving {i} frames to {save_dir}.')

def process_all_videos(input_dir, output_dir):
    for video_path in glob.glob(os.path.join(input_dir, '*.mp4')):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join(output_dir, video_name)
        capture_frames(video_path, save_dir)

# 실행 예
process_all_videos('./video_sample', './pilot_study/separated_samples')


