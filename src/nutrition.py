import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class NutritionPer100g:
    calories: float
    protein: float
    fat: float
    carbs: float

def load_nutrition_db(path: str):
    df = pd.read_csv(path)
    df["food_name"] = df["food_name"].str.strip().str.lower()
    return df

def get_nutrition_for(food_name: str, df):
    row = df.loc[df["food_name"] == food_name.lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    return NutritionPer100g(
        calories=float(r["calories_kcal_100g"]),
        protein=float(r["protein_g_100g"]),
        fat=float(r["fat_g_100g"]),
        carbs=float(r["carbs_g_100g"]),
    )

def scale_per_serving(nutri_100g: NutritionPer100g, grams: float) -> Dict[str, float]:
    factor = max(grams, 0.0) / 100.0
    return {
        "calories_kcal": round(nutri_100g.calories * factor, 2),
        "protein_g": round(nutri_100g.protein * factor, 2),
        "fat_g": round(nutri_100g.fat * factor, 2),
        "carbs_g": round(nutri_100g.carbs * factor, 2),
    }
