import argparse

import cv2
import mmcv
import torch
import pickle
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmengine.utils import ProgressBar
from moviepy import VideoFileClip, ImageSequenceClip
from mmyolo.registry import VISUALIZERS
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import os.path as osp

import supervision as sv
import itertools

import os

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World video demo')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('video', help='video file path')
    parser.add_argument('transcript', help='pkl transcript file path')
    parser.add_argument('fps', default=2, type=int)
    # parser.add_argument(
    #     'text',
    #     help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    # )
    parser.add_argument('--device',
                        default='cpu',
                        help='device used for inference')
    parser.add_argument('--score-thr',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    args = parser.parse_args()
    return args


def inference_detector(model, image, texts, test_pipeline, target_dir, image_name, score_thr=0.3):
    data_info = dict(img_id=0, img=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]
    pred_instances = pred_instances.cpu().numpy()
    return pred_instances.scores.mean() if len(pred_instances.scores > 0) else 0.0

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    cv2.imwrite(osp.join(target_dir, f'{image_name}.png'), image)
    return output


def extract_frames_between(video_path, start_time, end_time, fps=1):
    clip = VideoFileClip(video_path).subclipped(start_time, end_time)
    return list(clip.iter_frames(fps=fps))


def save_frames_as_video(frames, output_path, fps=1):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264')


def get_top_k(scores, k=5):
    # Sort by 'score' in descending order
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    return sorted_scores[:k]


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    model_blip2 = Blip2ForImageTextRetrieval.from_pretrained(
        "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
    processor_blip2 = AutoProcessor.from_pretrained(
        "Salesforce/blip2-itm-vit-g")
    model_blip2.to(args.device)
    # build test pipeline
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    # if args.text.endswith('.txt'):
    #     with open(args.text) as f:
    #         lines = f.readlines()
    #     texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    # else:
    #     texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # visualizer = VISUALIZERS.build(
    #     dict(type='DetLocalVisualizer', name='visualizer'))

    with open(args.transcript, "rb") as f:
        sentences = pickle.load(f)
    progress_bar = ProgressBar(len(sentences))
    for sentence in sentences:
        target_dir = osp.join(output_dir, sentence['text'])
        if not osp.exists(target_dir):
            os.makedirs(target_dir)
        texts = [[sentence['text']]]
        # reparameterize texts
        model.reparameterize(texts)
        visualizer.dataset_meta = dict(classes=texts, palette=None)
        video_reader = mmcv.VideoReader(args.video)
        original_fps = video_reader.fps
        start_frame = max(int(sentence["start"] * original_fps) - 5, 0)
        end_frame = min(
            int(sentence["end"] * original_fps) + 5, len(video_reader))
        target_fps = args.fps
        # e.g., if original=30, target=2 → step=15
        step = int(original_fps / target_fps)
        frame_indices = list(range(start_frame, end_frame, step))

        # frames = extract_frames_between(
        #     args.video, sentence["start"], sentence["end"], fps=args.fps)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(
        #     f"{sentence['text']}.mp4", fourcc,
        #     # video_reader.fps,
        #     args.fps,
        #     (video_reader.width, video_reader.height))
        frames = [video_reader[i]
                  for i in frame_indices if i < len(video_reader)]

        scores = []
        for i, frame in enumerate(frames):
            scores.append(dict(frame=frame,
                               score=inference_detector(model,
                                                        frame,
                                                        texts,
                                                        test_pipeline,
                                                        target_dir,
                                                        i,
                                                        score_thr=args.score_thr)))
        sorted_scores = get_top_k(scores=scores)
        if (sorted_scores[0]['score'] == 0.0):
            progress_bar.update()
            continue
        for item in sorted_scores[:]:
            if item['score'] == 0.0:
                continue
            inputs = processor_blip2(images=frame, text=list(itertools.chain(*texts))[0],
                                     return_tensors="pt").to(args.device, torch.float16)
            itm_out = model_blip2(**inputs, use_image_text_matching_head=True)
            logits_per_image = torch.nn.functional.softmax(
                itm_out.logits_per_image, dim=1)
            probs = logits_per_image.softmax(dim=1)
            if probs[0][0] < probs[0][1]:
                with open(f'{args.out}/script.txt', 'a') as f:
                    f.write(texts[0][0])

        progress_bar.update()

    # video_reader = mmcv.VideoReader(args.video)
    # video_writer = None
    # if args.out:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(
    #         args.out, fourcc,
    #         # video_reader.fps,
    #         3.0,
    #         (video_reader.width, video_reader.height))

    # for frame in track_iter_progress(video_reader):
    #     result = inference_detector(model,
    #                                 frame,
    #                                 texts,
    #                                 test_pipeline,
    #                                 score_thr=args.score_thr)
    #     visualizer.add_datasample(name='video',
    #                               image=frame,
    #                               data_sample=result,
    #                               draw_gt=False,
    #                               show=False,
    #                               pred_score_thr=args.score_thr)
    #     frame = visualizer.get_image()

    #     if args.out:
    #         video_writer.write(frame)

    # if video_writer:
    #     video_writer.release()


if __name__ == '__main__':
    main()
