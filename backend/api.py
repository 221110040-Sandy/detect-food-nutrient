import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from datetime import datetime, date
from typing import Optional

from src.model import build_model, load_checkpoint
from src.nutrition import (
    load_nutrition_db, get_nutrition_for, scale_per_serving,
    DailyTracker, DAILY_REQUIREMENTS
)

CKPT_PATH = "food_classifier_dataminds.pt"
NUTRITION_CSV = "data/nutrition_db.csv"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

img_tfm = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

state_dict, class_names, arch = load_checkpoint(CKPT_PATH, device=DEVICE)
model = build_model(num_classes=len(class_names), arch=arch)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

nutri_df = load_nutrition_db(NUTRITION_CSV)

# In-memory daily tracker storage (per session, akan reset jika server restart)
# Untuk production, gunakan database
daily_trackers: dict[str, DailyTracker] = {}


def get_or_create_tracker(date_str: str) -> DailyTracker:
    """Get atau buat tracker untuk tanggal tertentu"""
    if date_str not in daily_trackers:
        daily_trackers[date_str] = DailyTracker(date=date_str)
    return daily_trackers[date_str]


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
            fiber_g=per_portion["fiber_g"],
            sugar_g=per_portion["sugar_g"],
            sodium_mg=per_portion["sodium_mg"],
        ),
    )


# ============ DAILY TRACKER ENDPOINTS ============

class AddFoodRequest(BaseModel):
    food_name: str
    portion_g: float
    nutrition: NutritionBlock


class FoodEntryResponse(BaseModel):
    index: int
    food_name: str
    portion_g: float
    nutrition: NutritionBlock
    timestamp: str


class NutrientStatus(BaseModel):
    label: str
    current: float
    target: float
    type: str  # "min", "max", "around"
    status: str  # "kurang", "terpenuhi", "berlebih", "aman"
    message: str
    percentage: float


class DailyStatusResponse(BaseModel):
    date: str
    entries: list[FoodEntryResponse]
    total_nutrition: NutritionBlock
    nutrient_status: dict[str, NutrientStatus]
    summary: str


class DailyRequirementInfo(BaseModel):
    nutrient: str
    label: str
    target: float
    type: str
    description: str


@app.get("/daily-requirements")
async def get_daily_requirements():
    """Dapatkan info kebutuhan gizi harian"""
    requirements = []
    descriptions = {
        "min": "Minimal harus tercapai",
        "max": "Maksimal tidak boleh melebihi",
        "around": "Target kira-kira"
    }
    for nutrient, info in DAILY_REQUIREMENTS.items():
        requirements.append(DailyRequirementInfo(
            nutrient=nutrient,
            label=info["label"],
            target=info["target"],
            type=info["type"],
            description=descriptions.get(info["type"], "")
        ))
    return {"requirements": requirements}


@app.post("/tracker/add")
async def add_food_to_tracker(
    request: AddFoodRequest,
    date_str: str = Query(default=None, description="Tanggal (YYYY-MM-DD), default hari ini")
):
    """Tambahkan makanan ke tracker harian"""
    if date_str is None:
        date_str = date.today().isoformat()
    
    tracker = get_or_create_tracker(date_str)
    timestamp = datetime.now().strftime("%H:%M")
    
    nutrition_dict = {
        "calories_kcal": request.nutrition.calories_kcal,
        "protein_g": request.nutrition.protein_g,
        "fat_g": request.nutrition.fat_g,
        "carbs_g": request.nutrition.carbs_g,
        "fiber_g": request.nutrition.fiber_g,
        "sugar_g": request.nutrition.sugar_g,
        "sodium_mg": request.nutrition.sodium_mg,
    }
    
    tracker.add_entry(
        food_name=request.food_name,
        portion_g=request.portion_g,
        nutrition=nutrition_dict,
        timestamp=timestamp
    )
    
    return {"success": True, "message": f"Berhasil menambahkan {request.food_name} ({request.portion_g}g)"}


@app.delete("/tracker/remove/{index}")
async def remove_food_from_tracker(
    index: int,
    date_str: str = Query(default=None, description="Tanggal (YYYY-MM-DD), default hari ini")
):
    """Hapus makanan dari tracker berdasarkan index"""
    if date_str is None:
        date_str = date.today().isoformat()
    
    tracker = get_or_create_tracker(date_str)
    
    if tracker.remove_entry(index):
        return {"success": True, "message": f"Berhasil menghapus item #{index}"}
    else:
        raise HTTPException(status_code=404, detail=f"Entry dengan index {index} tidak ditemukan")


@app.get("/tracker/status", response_model=DailyStatusResponse)
async def get_daily_status(
    date_str: str = Query(default=None, description="Tanggal (YYYY-MM-DD), default hari ini")
):
    """Dapatkan status gizi harian beserta analisis kebutuhan yang kurang/lebih"""
    if date_str is None:
        date_str = date.today().isoformat()
    
    tracker = get_or_create_tracker(date_str)
    
    # Build entries response
    entries = []
    for i, entry in enumerate(tracker.entries):
        entries.append(FoodEntryResponse(
            index=i,
            food_name=entry.food_name,
            portion_g=entry.portion_g,
            nutrition=NutritionBlock(**entry.nutrition),
            timestamp=entry.timestamp
        ))
    
    # Get totals
    totals = tracker.get_total_nutrition()
    
    # Get remaining needs analysis
    needs = tracker.get_remaining_needs()
    nutrient_status = {
        k: NutrientStatus(**v) for k, v in needs.items()
    }
    
    # Get summary text
    summary = tracker.get_summary_text()
    
    return DailyStatusResponse(
        date=date_str,
        entries=entries,
        total_nutrition=NutritionBlock(**totals),
        nutrient_status=nutrient_status,
        summary=summary
    )


@app.post("/tracker/clear")
async def clear_tracker(
    date_str: str = Query(default=None, description="Tanggal (YYYY-MM-DD), default hari ini")
):
    """Hapus semua entry untuk tanggal tertentu"""
    if date_str is None:
        date_str = date.today().isoformat()
    
    if date_str in daily_trackers:
        daily_trackers[date_str] = DailyTracker(date=date_str)
    
    return {"success": True, "message": f"Tracker untuk {date_str} sudah direset"}
