# Databricks notebook source
!pip install opencv-python==4.8.0.76
!pip install Pillow==9.0.0
!pip install timm


!pip install "/dbfs/mnt/cep-refresh-2023-exploration/praneeth/prod_detect/detectron2-0.6-cp39-cp39-linux_x86_64.whl"
%pip install git+https://github.com/openai/CLIP.git
!pip install --upgrade numpy
# !pip install scikit-image
!pip install diffusers transformers==4.32.0 
!pip install torch==1.13.1 torchvision==0.14.1
  

# COMMAND ----------

import cv2
import os
import sys
from PIL import Image
import tqdm
import math



## Product detection
root_folder_path = #Give the path to the folder for prod_detect ex:- "/dbfs/mnt/cep-refresh-2023-exploration/praneeth/"
product_detection_utilities = os.path.join(root_folder_path,"prod_detect")
path2detic = os.path.join(product_detection_utilities, "Detic")

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
sys.path.append(path2detic)
sys.path.append(path2detic + '/third_party/CenterNet2')

from detic.modeling.text.text_encoder import build_text_encoder
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInpaintPipeline 

from transformers import BlipProcessor, BlipForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")

# COMMAND ----------

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(os.path.join(path2detic,"configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"))
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
predictor = DefaultPredictor(cfg)

# COMMAND ----------

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    # texts = [prompt + x if x!="Person" else "a Person" for x in vocabulary]
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def detect_product(im, product_keywords, threshold=0.4):
    if '__test' in ([x for x in MetadataCatalog.keys()]):
        MetadataCatalog.pop('__test')
    metadata = MetadataCatalog.get("__test")
    metadata.thing_classes = product_keywords
    vocabulary = 'custom'
    classifier = get_clip_embeddings(metadata.thing_classes)
    num_classes = len(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, num_classes)
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = threshold

    outputs = predictor(im)
    result_pd = {}
    temp = outputs['instances']._fields['pred_classes'].cpu().numpy()

    result_pd = {
        'pred_boxes': np.round(outputs['instances']._fields['pred_boxes'].tensor.cpu().numpy()),
        'pred_classes': np.array(metadata.thing_classes)[temp.astype(int)],
        'pred_scores': outputs['instances']._fields['scores'].cpu().numpy(),
        'pred_masks': outputs['instances'].pred_masks.cpu().numpy()
    }
    return result_pd




def create_masked_image(image, result_pd):
    # Create a combined mask from the predicted masks
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for mask in result_pd['pred_masks']:
        combined_mask = cv2.bitwise_or(combined_mask, mask.astype(np.uint8) * 255)

    # Invert the combined mask to have the required parts in black and the rest in white
    inpainting_mask = cv2.bitwise_not(combined_mask)

    return inpainting_mask

def blip_img_captions(image_path):
    try:
        raw_image = Image.open(image_path)
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        image_caption = processor.decode(out[0], skip_special_tokens=True)
    except:
        image_caption= ""     
    return   image_caption


def get_ai_generated_img(focus_item,masked_img_path,image_name,caption_frm_img):

    # input_image = Image.open(focus_item).convert("RGB")
    # mask_image = Image.open(masked_img_path + image_name + "_masked.jpg").convert("L")
    # mask_image = mask_image.resize(input_image.size)

    input_image_cv = cv2.imread(focus_item)
    mask_image_cv = cv2.imread(masked_img_path + image_name + "_masked.jpg", cv2.IMREAD_GRAYSCALE)

    mask_image_cv = cv2.resize(mask_image_cv, (input_image_cv.shape[1], input_image_cv.shape[0]))


    input_image_cv_rgb = cv2.cvtColor(input_image_cv, cv2.COLOR_BGR2RGB)

    input_image = Image.fromarray(input_image_cv_rgb)
    mask_image = Image.fromarray(mask_image_cv)
    input_width, input_height = input_image.size

    input_width = int(math.ceil(input_width / 8.0)) * 8
    input_height = int(math.ceil(input_height / 8.0)) * 8

    # print("Input Image Size: ", input_image.size)
    # print("Mask Image Size: ", mask_image.size)

    # prompt =  "create a portrait,clear face features, ultrarealistic ,cinematic lighting, award winning photo, 8 k, hi res –testp –ar 3:4 –upbeta  smooth, sharp focus, high resolution Concept is :" + caption_frm_img  

    # negative_prompt = "disfigured, ugly, bad,face shapeout ,immature, 3d, b&w" 

    prompt = "create a AI art portrait, , ultra-realistic, cinematic lighting, award-winning photo, 8K, high resolution, smooth, sharp focus, high resolution, beautiful, detailed, vibrant, professional photography, photorealistic, masterpiece, dramatic lighting, intricate details Concept is: " + caption_frm_img
    

    negative_prompt = "disfigured, ugly, bad, face shapeout, immature, 3d, b&w, low resolution, blurry, grainy, poorly lit, out of focus, pixelated cartoonish, exaggerated, overexposed, underexposed, unnatural"
    

    strength = 0.6
    guidance_scale = 15
    num_inference_steps = 100

    result = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=mask_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,    #height = input_height,
   # width = input_width,
    ).images[0]

    # print("Generated Image Size: ", result.size)


    return result



# COMMAND ----------

masked_img_path = root_folder_path + "/Masked_image/"
output_img_path = root_folder_path + "/Generated_image/"
os.makedirs(masked_img_path, exist_ok=True)
os.makedirs(output_img_path, exist_ok=True)
keywords = ["Shoe","Sneaker","Bottle","Cup","Sandal","Perfume","Toy","Sunglasses","Car","Water Bottle","Chair","Office Chair", "Can", "Cap", "Hat","Couch","Wristwatch","Glass","Bag","Handbag","Baggage","Suitcase","Headphones","Jar","Vase"]

# COMMAND ----------

img_path = root_folder_path + "/prod_detect/ztest_sample_img/"
img_lst = os.listdir(img_path) 
# img_lst

for image_name in tqdm.tqdm(img_lst[3:5]):
    
    focus_item = img_path + image_name
    img = cv2.imread(focus_item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


    result_pd = detect_product(img, keywords)
    masked_image = create_masked_image(img, result_pd)
    cv2.imwrite(masked_img_path + image_name + "_masked.jpg", masked_image)
    plt.imshow(masked_image)
    plt.axis("off")
    plt.show()


    caption_frm_img = blip_img_captions(focus_item)
    gernareted_img = get_ai_generated_img(focus_item,masked_img_path,image_name,caption_frm_img)
    # cv2.imwrite(output_img_path + image_name + "_AI_generated.jpg", gernareted_img)
    plt.imshow(gernareted_img)
    plt.axis("off")
    plt.savefig(output_img_path + image_name + "_AI_generated.jpg")
    plt.show()
    


# COMMAND ----------

result_pd
