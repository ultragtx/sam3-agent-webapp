import torch
import numpy as np
import matplotlib.pyplot as plt
#################################### For Image ####################################
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model(
    bpe_path="essential_assets/bpe_simple_vocab_16e6.txt.gz",
)
processor = Sam3Processor(model)
# # Load an image
# image = Image.open("./sam3/assets/images/truck.jpg")
image = Image.open("sample.jpg")
inference_state = processor.set_image(image)
# # Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="the 2nd elephant from left")


def save_sam3_masks_and_boxes(image, masks, boxes, scores, save_path, alpha=0.4, score_thresh=0.5):
    image_np = np.array(image).astype(float)

    # 叠加 mask
    for mask in masks:
        mask_np = mask.cpu().numpy().squeeze()
        red_layer = np.zeros_like(image_np)
        red_layer[:, :, 0] = mask_np * 255

        image_np = np.where(mask_np[..., None] > 0.5,
                            image_np * (1 - alpha) + red_layer * alpha,
                            image_np)

    # 回到 PIL
    out = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(out)

    # 画 box
    for i in range(len(boxes)):
        if scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
        draw.text((x1, y1), f"{scores[i]:.2f}", fill="yellow")

    out.save(save_path)
# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
save_sam3_masks_and_boxes(image, masks, boxes, scores, 'demo.jpg')
