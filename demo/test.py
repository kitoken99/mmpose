# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
import glob

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# module for time
from datetime import datetime

# global var
OUTPUT_ROOT ="movies/outputs"
INPUT_ROOT ="movies/inputs"
BASE_MIN = 5

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    timestamp_begin = time.time()

    # 引数
    parser = ArgumentParser()
    parser.add_argument('--det_config', default="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py", help='Config file for detection')
    parser.add_argument('--det_checkpoint', default="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth", help='Checkpoint file for detection')
    parser.add_argument('--pose_config', default="configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py", help='Config file for pose')
    parser.add_argument('--pose_checkpoint', default="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth", help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, default='', help='Image/Video file')
    parser.add_argument('--show',action='store_true',default=False,help='whether to show img')
    parser.add_argument('--save-predictions',action='store_true',default=False,help='whether to save predicted results')
    parser.add_argument('--det-cat-id',type=int,default=0,help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr',type=float,default=0.3,help='Bounding box score threshold')
    parser.add_argument('--nms-thr',type=float,default=0.3,help='IoU threshold for bounding box NMS')
    parser.add_argument('--kpt-thr',type=float,default=0.3,help='Visualizing keypoint thresholds')
    parser.add_argument('--draw-heatmap',action='store_true',default=False,help='Draw heatmap predicted by the model')
    parser.add_argument('--show-kpt-idx',action='store_true',default=False,help='Whether to show the index of keypoints')
    parser.add_argument('--skeleton-style',default='mmpose',type=str,choices=['mmpose', 'openpose'],help='Skeleton style selection')
    parser.add_argument('--radius',type=int,default=3,help='Keypoint radius for visualization')
    parser.add_argument('--thickness',type=int,default=1,help='Link thickness for visualization')
    parser.add_argument('--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument('--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument('--draw-bbox', action='store_true', help='Draw bboxes of instances')

    # エラーチェック
    assert has_mmdet, 'Please install mmdet to run the demo.'
    args = parser.parse_args()

    global OUTPUT_ROOT
    global INPUT_ROOT

    # ファイル処理
    subjects = glob.glob("./"+INPUT_ROOT+"/*")
    for subject in subjects:
        print(subject)
    input_back = INPUT_ROOT + "/" + args.input + "_back.mp4"
    input_side = INPUT_ROOT + "/" + args.input + "_side.mp4"

    mmengine.mkdir_or_exist(OUTPUT_ROOT)
    output_file_back = os.path.join(OUTPUT_ROOT, os.path.basename(input_back))
    output_file_side = os.path.join(OUTPUT_ROOT, os.path.basename(input_side))

    if args.save_predictions:
        args.back_pred_save_path = f'{OUTPUT_ROOT}/results_' \
            f'{os.path.splitext(os.path.basename(input_back))[0]}.json'
        args.side_pred_save_path = f'{OUTPUT_ROOT}/results_' \
            f'{os.path.splitext(os.path.basename(input_side))[0]}.json'

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)


    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)
    
    
    # inference
    print_log(
            f'the inference has been started',
            logger='current',
            level=logging.INFO)


    cap_back = cv2.VideoCapture(input_back)
    cap_side = cv2.VideoCapture(input_side)

    video_writer_back = None
    video_writer_side = None
    pred_instances_back_list = []
    pred_instances_side_list = []
    frame_idx = 0
    global BASE_MIN
    while cap_back.isOpened() and cap_side.isOpened and frame_idx < 30*60*BASE_MIN:
        success_back, frame_back = cap_back.read()
        success_side, frame_side = cap_side.read()
        frame_idx += 1

        if not success_back:
            break
        if not success_side:
            break
        # topdown pose estimation
        pred_instances_back = process_one_image(args, frame_back, detector, pose_estimator, visualizer, 0.001)
        if args.save_predictions:
            # save prediction results
            pred_instances_back_list.append(
                dict(
                    frame_id=frame_idx,
                    instances=split_instances(pred_instances_back)))
        if output_file_back:
            frame_vis_back = visualizer.get_image()
            if video_writer_back is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # the size of the image with visualization may vary
                # depending on the presence of heatmaps
                video_writer_back = cv2.VideoWriter(
                    output_file_back,
                    fourcc,
                    30,  # saved fps
                    (frame_vis_back.shape[1], frame_vis_back.shape[0]))
            video_writer_back.write(mmcv.rgb2bgr(frame_vis_back))

        

        pred_instances_side = process_one_image(args, frame_side, detector, pose_estimator, visualizer, 0.001)

        if(frame_idx%(30*10) == 0):
            print_log(
                f'{frame_idx/30} seconds has done',
                logger='current',
                level=logging.INFO)
        # print_log(
        #     f'{frame_idx} frames has done',
        #     logger='current',
        #     level=logging.INFO)

        if args.save_predictions:
            # save prediction results
            pred_instances_side_list.append(
                dict(
                    frame_id=frame_idx,
                    instances=split_instances(pred_instances_side)))

        # output videos
        if output_file_side:
            frame_vis_side = visualizer.get_image()
            if video_writer_side is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # the size of the image with visualization may vary
                # depending on the presence of heatmaps
                video_writer_side = cv2.VideoWriter(
                    output_file_side,
                    fourcc,
                    30,  # saved fps
                    (frame_vis_side.shape[1], frame_vis_side.shape[0]))

            video_writer_side.write(mmcv.rgb2bgr(frame_vis_side))

    if video_writer_back:
        video_writer_back.release()
    if video_writer_side:
        video_writer_side.release()

    cap_back.release()
    cap_side.release()



    timestamp_end = time.time()
    inference_time = int(timestamp_end-timestamp_begin)

    print_log(
            f'the inference has been finished in {inference_time} seconds',
            logger='current',
            level=logging.INFO)

    if args.save_predictions:
        with open(args.back_pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_back_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.back_pred_save_path}')
        with open(args.side_pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_side_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.side_pred_save_path}')

    if output_file_back:
        print_log(
            f'the output video has been saved at {output_file_back}',
            logger='current',
            level=logging.INFO)
    if output_file_side:
        print_log(
            f'the output video has been saved at {output_file_side}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
    