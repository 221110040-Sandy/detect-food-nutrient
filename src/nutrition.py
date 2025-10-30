import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NutritionPer100g:
    calories: float
    protein: float
    fat: float
    carbs: float
    fiber: float
    sugar: float
    sodium_mg: float


def load_nutrition_db(path: str):
    df = pd.read_csv(path)
    df["food_name"] = df["food_name"].str.strip().str.lower()
    return df


def get_nutrition_for(food_name: str, df) -> Optional[NutritionPer100g]:
    row = df.loc[df["food_name"] == food_name.lower()]
    if row.empty:
        return None
    r = row.iloc[0]
    return NutritionPer100g(
        calories=float(r["calories_kcal_100g"]),
        protein=float(r["protein_g_100g"]),
        fat=float(r["fat_g_100g"]),
        carbs=float(r["carbs_g_100g"]),
        fiber=float(r.get("fiber_g_100g", 0.0)),
        sugar=float(r.get("sugar_g_100g", 0.0)),
        sodium_mg=float(r.get("sodium_mg_100g", 0.0)),
    )


def scale_per_serving(nutri_100g: NutritionPer100g, grams: float) -> Dict[str, float]:
    factor = max(grams, 0.0) / 100.0
    return {
        "calories_kcal": round(nutri_100g.calories * factor, 2),
        "protein_g": round(nutri_100g.protein * factor, 2),
        "fat_g": round(nutri_100g.fat * factor, 2),
        "carbs_g": round(nutri_100g.carbs * factor, 2),
        "fiber_g": round(nutri_100g.fiber * factor, 2),
        "sugar_g": round(nutri_100g.sugar * factor, 2),
        "sodium_mg": round(nutri_100g.sodium_mg * factor, 2),
    }
