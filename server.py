from fastapi import FastAPI, UploadFile, Form
from groundingdino.util.inference import load_model, load_image, predict
import torch
import tempfile

app = FastAPI()

# 🔒 Load model ONCE at startup (CPU)
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)
model = model.to("cpu")
model.eval()


@app.post("/predict")
async def predict_api(
    image: UploadFile,
    text: str = Form(...)
):
    # Save uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(await image.read())
        img_path = f.name

    # GroundingDINO helper → returns PIL + tensor
    image_pil, image_tensor = load_image(img_path)

    # 🔥 THIS IS THE FIX (tensor → cpu)
    image_tensor = image_tensor.to("cpu")

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text,
        box_threshold=0.35,
        text_threshold=0.25,
        device="cpu"
    )

    return {
        "boxes": boxes.tolist(),
        "phrases": phrases
    }

