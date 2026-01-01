import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import date


# Rekomendasi kebutuhan gizi harian (AKG Indonesia)
DAILY_REQUIREMENTS = {
    "calories_kcal": {"target": 2000, "type": "around", "label": "Energi"},
    "protein_g": {"target": 50, "type": "around", "label": "Protein"},
    "fat_g": {"target": 65, "type": "max", "label": "Lemak Total"},
    "carbs_g": {"target": 275, "type": "around", "label": "Karbohidrat"},
    "fiber_g": {"target": 25, "type": "min", "label": "Serat"},
    "sugar_g": {"target": 50, "type": "max", "label": "Gula"},
    "sodium_mg": {"target": 2000, "type": "max", "label": "Natrium"},
}


@dataclass
class FoodEntry:
    """Satu entry makanan yang dikonsumsi"""
    food_name: str
    portion_g: float
    nutrition: Dict[str, float]
    timestamp: str = ""


@dataclass
class DailyTracker:
    """Tracker untuk konsumsi harian"""
    date: str
    entries: List[FoodEntry] = field(default_factory=list)
    
    def add_entry(self, food_name: str, portion_g: float, nutrition: Dict[str, float], timestamp: str = ""):
        entry = FoodEntry(
            food_name=food_name,
            portion_g=portion_g,
            nutrition=nutrition,
            timestamp=timestamp
        )
        self.entries.append(entry)
        return entry
    
    def remove_entry(self, index: int) -> bool:
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            return True
        return False
    
    def get_total_nutrition(self) -> Dict[str, float]:
        """Hitung total gizi dari semua makanan hari ini"""
        totals = {
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0,
            "fiber_g": 0.0,
            "sugar_g": 0.0,
            "sodium_mg": 0.0,
        }
        for entry in self.entries:
            for key in totals:
                totals[key] += entry.nutrition.get(key, 0.0)
        # Round semua nilai
        return {k: round(v, 2) for k, v in totals.items()}
    
    def get_remaining_needs(self) -> Dict[str, Dict]:
        """
        Hitung kebutuhan gizi yang masih kurang/lebih hari ini.
        Returns dict dengan status per nutrisi.
        """
        totals = self.get_total_nutrition()
        result = {}
        
        for nutrient, req in DAILY_REQUIREMENTS.items():
            current = totals.get(nutrient, 0.0)
            target = req["target"]
            req_type = req["type"]
            label = req["label"]
            
            if req_type == "min":
                # Minimal harus mencapai target (misal serat >= 25g)
                remaining = max(0, target - current)
                if current >= target:
                    status = "terpenuhi"
                    message = f"✅ {label} sudah tercukupi ({current:.1f}/{target}g)"
                else:
                    status = "kurang"
                    message = f"⚠️ {label} masih kurang {remaining:.1f}g lagi"
            
            elif req_type == "max":
                # Maksimal tidak boleh lebih dari target (misal gula <= 50g)
                remaining = max(0, target - current)
                if current <= target:
                    status = "aman"
                    message = f"✅ {label} masih dalam batas ({current:.1f}/{target})"
                else:
                    over = current - target
                    status = "berlebih"
                    message = f"🚫 {label} sudah BERLEBIH {over:.1f} dari batas!"
            
            else:  # "around" - target kira-kira
                diff = target - current
                percentage = (current / target) * 100 if target > 0 else 0
                
                if percentage < 80:
                    status = "kurang"
                    message = f"⚠️ {label} baru {percentage:.0f}% (kurang ~{diff:.1f})"
                elif percentage <= 120:
                    status = "terpenuhi"
                    message = f"✅ {label} sudah cukup ({current:.1f}/{target})"
                else:
                    over = current - target
                    status = "berlebih"
                    message = f"⚠️ {label} berlebih {over:.1f} dari target"
            
            result[nutrient] = {
                "label": label,
                "current": current,
                "target": target,
                "type": req_type,
                "status": status,
                "message": message,
                "percentage": round((current / target) * 100, 1) if target > 0 else 0
            }
        
        return result
    
    def get_summary_text(self) -> str:
        """Generate ringkasan kebutuhan dalam format text"""
        needs = self.get_remaining_needs()
        lines = ["📊 RINGKASAN KEBUTUHAN GIZI HARI INI", "=" * 40]
        
        # Group by status
        kurang = []
        berlebih = []
        terpenuhi = []
        
        for nutrient, info in needs.items():
            if info["status"] == "kurang":
                kurang.append(info["message"])
            elif info["status"] == "berlebih":
                berlebih.append(info["message"])
            else:
                terpenuhi.append(info["message"])
        
        if kurang:
            lines.append("\n🔴 PERLU DITAMBAH:")
            lines.extend(kurang)
        
        if berlebih:
            lines.append("\n🟡 SUDAH BERLEBIH:")
            lines.extend(berlebih)
        
        if terpenuhi:
            lines.append("\n🟢 SUDAH TERCUKUPI:")
            lines.extend(terpenuhi)
        
        return "\n".join(lines)


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
