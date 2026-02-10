import gradio as gr
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image, ImageDraw
import easyocr
import numpy as np

import os


os.environ["TRANSFORMERS_CACHE"] = "X:/Programming/Models"
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)

vlm = AutoModelForVision2Seq.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

ocr = easyocr.Reader(['ru', 'en'], gpu=True)

def run_vlm(image: Image.Image):
    prompt = """
Extract ONLY sensitive text from the image.

Rules:
- Output ONLY text that appears verbatim in the image
- Sensitivity must depend on visual context
- One item per line
- No explanations
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt"
    ).to(vlm.device)

    outputs = vlm.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    objects = []

    for item in processor.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1].split('\n'):
        objects.extend(item.split())

    objects = [item for item in objects if len(item) > 3]

    return objects

def run_ocr(image_np):
    output_ocr = ocr.readtext(image_np, low_text=0.3, text_threshold=0.5, canvas_size=3200)

    items = []
    for line in output_ocr:
        bbox = line[0]
        text = line[1]
        score = line[2]
        items.append({
            "text": text,
            "bbox": bbox,
            "score": score
        })

    return items


def ground_phrases(phrases, ocr_items):
    grounded = []

    for phrase in phrases:
        for item in ocr_items:
            if phrase.lower() in item["text"].lower():
                grounded.append({
                    "text": phrase,
                    "bbox": item["bbox"]
                })

    return grounded

def redact_image(image, grounded_boxes):
    redacted = image.copy()
    draw = ImageDraw.Draw(redacted)
    
    for item in grounded_boxes:
        bbox = item['bbox']
        
        points = []
        for point in bbox:
            points.extend([point[0], point[1]])

        fill_color = (0, 0, 0, 64)
        outline_color = (0, 0, 0)
        
        draw.polygon(points, fill=fill_color, outline=outline_color, width=8)
    
    return redacted

def process_image(image: Image.Image):
    if image is None:
        return None, "No image uploaded"

    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    sensitive_phrases = run_vlm(image_rgb)

    if not sensitive_phrases:
        return image_rgb, "No sensitive text detected"

    ocr_items = run_ocr(image_np)

    grounded = ground_phrases(sensitive_phrases, ocr_items)

    if not grounded:
        return image_rgb, "Sensitive text found, but grounding failed"

    redacted_img = redact_image(image, grounded)

    report = "Redacted items:\n" + "\n".join(
        f"- {item}" for item in set([item['text'] for item in grounded])
    )

    return redacted_img, report


with gr.Blocks(title="Multimodal Sensitive Data Redaction") as demo:
    gr.Markdown(
    """
# 🔒 Multimodal Sensitive Data Redaction

This service uses:
- **Vision–Language Model (VLM)** for context-aware sensitive text detection
- **OCR** for spatial grounding

Upload an image with diagrams or infrastructure schemes.
"""
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload image")
        output_image = gr.Image(type="pil", label="Redacted image")

    output_text = gr.Textbox(
        label="Detection report",
        lines=10
    )

    run_button = gr.Button("Run redaction")

    run_button.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)