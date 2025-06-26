import itertools
import supervision as sv
import os
import cv2
import argparse
import os.path as osp
import numpy as np

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import re

import base64
import ollama

# Local LLM
ollama_llm = "llama3.2-vision:latest"
host = "https://c8c1-34-125-220-82.ngrok-free.app/"
client = ollama.Client(host=host)
model_name = 'openai/clip-vit-base-patch32'
clip_model = CLIPModel.from_pretrained(model_name).to('cpu')
clip_processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
clip_config = CLIPTextConfig.from_pretrained(model_name)
clip_model_proj = CLIPTP.from_pretrained(model_name, config=clip_config)

def encode_clip(model, processor, texts, device="cpu"):
    """Encode texts into CLIP embeddings."""
    num_per_batch = [len(t) for t in texts]
    text = list(itertools.chain(*texts))
    text = tokenizer(text=text, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        txt_outputs = clip_model_proj(**text)
    txt_feats = txt_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                  txt_feats.shape[-1])
    return txt_feats


def encode_clip_images(model, processor, pil_images, device="cpu"):
    """Encode PIL cropped images into CLIP embeddings."""
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        img_embeds = model.get_image_features(**inputs)
    img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
    return img_embeds


def rerank_yolo_results(detections, texts, pil_image, model_clip, processor_clip, device="cpu"):
    """
    Args:
        detections: List of YOLO results (dict), e.g.
            {"box": [x1, y1, x2, y2], "label": str, "orig_score": float}
        texts: List of text prompts
        pil_image: PIL.Image of the full scene
    Returns:
        List of results with added `similarity_score` and `final_score`.
    """
    text_embeds = encode_clip(model_clip, processor_clip, texts, device)
    cropped_images = []
    for det in detections:
        box = [int(b) for b in det["bboxes"][0]]
        cropped_images.append(pil_image.crop(box))
    img_embeds = encode_clip_images(
        model_clip, processor_clip, cropped_images, device)
    results = []
    print(text_embeds)
    for det, img_embed in zip(detections, img_embeds):
        sim_score = F.cosine_similarity(
            img_embed.unsqueeze(0), text_embeds[0], dim=-1)
        best_score, best_index = sim_score.max(dim=-1)

        results.append({
            **det,
            "similarity_score": best_score.item(),
            "final_score": (det["scores"] + best_score.item()) / 2,
            "matched_label": texts[best_index]
        })

    # Sắp xếp kết quả theo final_score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return torch.tensor(results)


def get_union_bbox(sub_bbox, obj_bbox):
    # Calculate the smallest bounding box that contains both subject and object
    x1 = min(sub_bbox[0], obj_bbox[0])
    y1 = min(sub_bbox[1], obj_bbox[1])
    x2 = max(sub_bbox[2], obj_bbox[2])
    y2 = max(sub_bbox[3], obj_bbox[3])
    return [x1, y1, x2, y2]

class AdapterAttention(nn.Module):
    """Adapter Attention ngoài YOLO‑World."""
    def __init__(self, embed_dim, num_heads=4):
        super(AdapterAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           batch_first=True)

    def forward(self, box_embs, text_embs):
        """box_embs: [N_boxes, embed_dim]
           text_embs: [N_queries, embed_dim]
        Returns:
           attention_weights: [N_boxes, N_queries]
        """
        box_embs = F.normalize(box_embs, dim=-1)
        text_embs = F.normalize(text_embs, dim=-1)

        box_seq = box_embs.unsqueeze(0)  # [1, N_boxes, embed_dim]
        text_seq = text_embs.unsqueeze(0)  # [1, N_queries, embed_dim]

        _, attn_weights = self.attn(box_seq, text_seq, text_seq)
        return attn_weights.squeeze(0)  # [N_boxes, N_queries]

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()

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

# blip_processor = Blip2Processor.from_pretrained(
#     "Salesforce/blip2-opt-2.7b")
# blip_model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b")


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.')
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False):
    data_info = dict(img_id=0, img_path=image, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
    print(pred_instances)
    # return
    rgb_image = Image.open(image_path).convert("RGB")
    final_results = rerank_yolo_results(
        pred_instances, texts, rgb_image, clip_model, clip_processor)
    print(final_results)
    final_results = final_results[final_results.final_score.float() >
                                    score_thr]

    if len(final_results.scores) > max_dets:
        indices = final_results.scores.float().topk(max_dets)[1]
        final_results = final_results[indices]

    final_results = final_results.cpu().numpy()

    if 'masks' in final_results:
        masks = final_results['masks']
    else:
        masks = None
    detections = sv.Detections(xyxy=final_results['bboxes'],
                               class_id=final_results['labels'],
                               confidence=final_results['final_score'],
                               mask=masks)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    image = cv2.imread(image_path)
    anno_image = image.copy()
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image)

    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(image_path)] = anno_image
        annotations_dict[osp.basename(image_path)] = detections

        ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=texts, images=images_dict,
            annotations=annotations_dict).as_yolo(
                annotations_directory_path=ANNOTATIONS_DIRECTORY,
                min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
                max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
                approximation_percentage=APPROXIMATION_PERCENTAGE)

    if show:
        cv2.imshow('Image', image)  # Provide window name
        k = cv2.waitKey(0)
        if k == 27:
            # wait for ESC key to exit
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)
    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    # reparameterize texts
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for image_path in images:
        inference_detector(model,
                           image_path,
                           texts,
                           test_pipeline,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation)
        progress_bar.update()
