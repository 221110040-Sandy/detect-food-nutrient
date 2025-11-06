import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

from src.model import build_model, load_checkpoint
from src.nutrition import load_nutrition_db, get_nutrition_for, scale_per_serving

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

val_tfm = transforms.Compose([
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_trained_model(ckpt_path: str):
    state_dict, class_names = load_checkpoint(ckpt_path, device=DEVICE)
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, class_names

def predict_food(model, class_names, img_path: str):
    img = Image.open(img_path).convert("RGB")
    x = val_tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    order = probs.argsort()[::-1]
    top_label = class_names[order[0]]
    top_conf  = float(probs[order[0]])
    top5 = [(class_names[i], float(probs[i])) for i in order[:5]]
    return top_label, top_conf, top5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path ke gambar makanan")
    parser.add_argument("--portion", type=float, default=250.0, help="porsi gram (default 250g)")
    parser.add_argument("--ckpt", default="food_classifier_dataminds.pt", help="checkpoint model")
    parser.add_argument("--nutdb", default="data/nutrition_db.csv", help="csv nutrisi")
    args = parser.parse_args()

    model, class_names = load_trained_model(args.ckpt)
    label, conf, topk = predict_food(model, class_names, args.image)
    print("Prediksi utama:", label, f"({conf*100:.1f}%)")
    print("Top-5:")
    for name, p in topk:
        print(f" - {name}: {p*100:.1f}%")

    nut_df = load_nutrition_db(args.nutdb)
    nut = get_nutrition_for(label, nut_df)
    if nut is None:
        print(f"[!] Gizi untuk '{label}' belum ada di CSV {args.nutdb}. Tambahkan dulu.")
        return

    per_serv = scale_per_serving(nut, args.portion)
    print(f"\nEstimasi gizi untuk porsi {args.portion} gram:")
    print(f"Kalori (kcal): {per_serv['calories_kcal']}")
    print(f"Protein (g):   {per_serv['protein_g']}")
    print(f"Lemak (g):     {per_serv['fat_g']}")
    print(f"Karbo (g):     {per_serv['carbs_g']}")
    print(f"Serat (g):     {per_serv['fiber_g']}")
    print(f"Gula (g):      {per_serv['sugar_g']}")
    print(f"Sodium (mg):   {per_serv['sodium_mg']}")

if __name__ == "__main__":
    main()
