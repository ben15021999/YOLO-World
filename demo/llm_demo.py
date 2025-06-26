import supervision as sv
import os
import cv2
import argparse
import os.path as osp
import numpy as np

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch

from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Local LLM
ollama_llm = "Llama3.1:latest"
llm = ChatOllama(model=ollama_llm, temperature=0)


def get_union_bbox(sub_bbox, obj_bbox):
    # Calculate the smallest bounding box that contains both subject and object
    x1 = min(sub_bbox[0], obj_bbox[0])
    y1 = min(sub_bbox[1], obj_bbox[1])
    x2 = max(sub_bbox[2], obj_bbox[2])
    y2 = max(sub_bbox[3], obj_bbox[3])
    return [x1, y1, x2, y2]

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()

ROLE_CANDIDATES = {
    "person": ["tennis player", "runner", "basketball player", "police officer"],
    "dog": ["service dog", "police dog", "pet"]
}


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

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

# --- Utils ---


def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    # return blip_processor.decode(output[0], skip_special_tokens=True)
    return blip_processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def ask_ollama(template):
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain


def classify_role(image_caption, caption, category, candidates):
    print(image_caption)
    print(caption)
    print(category)
    template = (
        f"I see this object with label '{category}' with caption: '{caption}'. The scene is described as: '{image_caption}'."
        f"Based on these information, tell me what is the most accurate role for this object or job if the object is a human or keep its label if you don't know?"
        f"Please provide the direct answer! Don't return any additional text"
    )
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({})


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
        pred_instances = pred_instances[pred_instances.scores.float() >
                                        score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    bboxes = pred_instances['bboxes']
    labels = pred_instances['labels']
    scores = pred_instances['scores']

    roles = []
    role_ids = []
    count = 0
    rgb_image = Image.open(image_path).convert("RGB")
    image_caption = generate_caption(rgb_image)
    for box, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(float, box)
        crop = rgb_image.crop((x1, y1, x2, y2))
        caption = generate_caption(crop)
        role = classify_role(image_caption, caption, texts[label][0], ROLE_CANDIDATES)
        roles.append(role)
        role_ids.append(count)
        count += 1

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=np.array(role_ids),
                               confidence=pred_instances['scores'],
                               mask=masks)
    labels = [
        f"{roles[class_id]} {confidence:0.2f}" for class_id, confidence in
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
