import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from torchvision import transforms

from src.model import build_model, load_checkpoint
from src.nutrition import load_nutrition_db, get_nutrition_for, scale_per_serving

CKPT_PATH = "food_classifier_dataminds.pt"
NUTRITION_CSV = "data/nutrition_db.csv"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

img_tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

state_dict, class_names = load_checkpoint(CKPT_PATH, device=DEVICE)
model = build_model(num_classes=len(class_names))
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

nutri_df = load_nutrition_db(NUTRITION_CSV)

class NutritionBlock(BaseModel):
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float

class PredictResponse(BaseModel):
    predicted_food: str
    confidence: float
    top5: list[tuple[str, float]]
    nutrition_per_100g: NutritionBlock | None
    nutrition_for_portion_g: NutritionBlock | None

app = FastAPI(title="DataMinds Nutrition API")

def run_inference(img_pil: Image.Image):
    x = img_tfm(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1][:5]
    top_label = class_names[order[0]]
    top_conf = float(probs[order[0]])
    top5 = [(class_names[i], float(probs[i])) for i in order]
    return top_label, top_conf, top5

@app.post("/predict", response_model=PredictResponse)
async def predict_food(
    file: UploadFile = File(...),
    portion_g: float = Query(250.0, ge=1.0, le=1000.0)
):
    try:
        img_bytes = await file.read()
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    label, conf, top5 = run_inference(img_pil)

    nutri100 = get_nutrition_for(label, nutri_df)
    if nutri100 is None:
        return PredictResponse(
            predicted_food=label,
            confidence=conf,
            top5=top5,
            nutrition_per_100g=None,
            nutrition_for_portion_g=None,
        )

    per_portion = scale_per_serving(nutri100, portion_g)

    return PredictResponse(
        predicted_food=label,
        confidence=conf,
        top5=top5,
        nutrition_per_100g=NutritionBlock(
            calories_kcal=nutri100.calories,
            protein_g=nutri100.protein,
            fat_g=nutri100.fat,
            carbs_g=nutri100.carbs,
            fiber_g=nutri100.fiber,
            sugar_g=nutri100.sugar,
            sodium_mg=nutri100.sodium_mg,
        ),
        nutrition_for_portion_g=NutritionBlock(
            calories_kcal=per_portion["calories_kcal"],
            protein_g=per_portion["protein_g"],
            fat_g=per_portion["fat_g"],
            carbs_g=per_portion["carbs_g"],
            fiber_g=per_portion.get("fiber_g", 0.0),
            sugar_g=per_portion.get("sugar_g", 0.0),
            sodium_mg=per_portion.get("sodium_mg", 0.0),
        ),
    )
