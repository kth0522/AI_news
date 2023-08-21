import cv2
import os
import csv
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
import time
import datetime

def process_image_folder(input_folder, output_directory, threshold=0.30):
    pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
    det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)

    # Check if output_directory exists, if not create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize the CSV file
    output_csv_file = os.path.join(output_directory, 'composition.csv')
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'human_id', 'composition'])

    # Create output directories for keypoints, body parts and pose images
    keypoints_dir = os.path.join(output_directory, 'keypoints')
    body_parts_dir = os.path.join(output_directory, 'body_parts')
    pose_images_dir = os.path.join(output_directory, 'pose_images')

    if not os.path.exists(keypoints_dir):
        os.makedirs(keypoints_dir)

    if not os.path.exists(body_parts_dir):
        os.makedirs(body_parts_dir)

    if not os.path.exists(pose_images_dir):
        os.makedirs(pose_images_dir)

    # Get a list of all image files and sort by frame number
    image_files = sorted(os.listdir(input_folder), key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))

    # Loop over all files in the input_folder
    for filename in image_files:
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # add any other image types here
            img = os.path.join(input_folder, filename)

            mmdet_results = inference_detector(det_model, img)

            person_results = process_mmdet_results(mmdet_results, cat_id=1)

            pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                           img,
                                                                           person_results,
                                                                           bbox_thr=0.3,
                                                                           format='xyxy',
                                                                           dataset=pose_model.cfg.data.test.type)

            vis_result = vis_pose_result(pose_model,
                                         img,
                                         pose_results,
                                         dataset=pose_model.cfg.data.test.type,
                                         show=False)

            # Get frame number from filename
            frame_number = os.path.splitext(filename)[0].split('_')[-1]

            # Save the pose result image
            pose_image_file = os.path.join(pose_images_dir, 'frame{}_pose.jpg'.format(frame_number))
            cv2.imwrite(pose_image_file, vis_result)

            # Process each human detected in the frame
            for human_id, pose_result in enumerate(pose_results):
                keypoints_file = os.path.join(keypoints_dir,
                                              'frame{}_human{}_keypoints.txt'.format(frame_number, human_id))
                body_parts_file = os.path.join(body_parts_dir,
                                               'frame{}_human{}_body_parts.txt'.format(frame_number, human_id))

                with open(keypoints_file, 'w') as f:
                    for item in pose_result['keypoints']:
                        f.write("%s\n" % item)

                body_part_bools = []
                with open(body_parts_file, 'w') as f:
                    for item in pose_result['keypoints']:
                        result = "True" if item[2] >= threshold else "False"
                        body_part_bools.append(result == "True")
                        f.write("%s\n" % result)

                head = any(body_part_bools[:5])
                upper_body = any(body_part_bools[5:11])
                full_body = any(body_part_bools[11:])

                composition = 'full body' if full_body and upper_body and head else 'upper body' if upper_body  and head else 'head' if head else 'none'

                with open(output_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_number, human_id, composition])


# Example usage
start = time.time()
process_image_folder('../news_data/separated_samples/ai_0001/', '../news_data/outputs/pose_estimation/ai_0001/')
end = time.time()
sec = (end - start)
result = datetime.timedelta(seconds=sec)
result_list = str(datetime.timedelta(seconds=sec)).split(".")
print("running time: ", result_list[0])
