import os
import cv2
import numpy as np

# === CONFIG ===
RGB_DIR = "extracted_frames_scene_2"
EVENT_MAP_DIR = "event_maps"
OUTPUT_DIR = "fused_output"
ALPHA = 0.7  # Strength of edge blending

os.makedirs(OUTPUT_DIR, exist_ok=True)

rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
event_files = sorted([f for f in os.listdir(EVENT_MAP_DIR) if f.endswith(".png")])

assert len(rgb_files) == len(event_files), "Mismatch in RGB and Event map count."

for rgb_name, event_name in zip(rgb_files, event_files):
    rgb_path = os.path.join(RGB_DIR, rgb_name)
    event_path = os.path.join(EVENT_MAP_DIR, event_name)

    # Load images
    rgb = cv2.imread(rgb_path)
    event = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

    # Resize event map to match RGB
    event = cv2.resize(event, (rgb.shape[1], rgb.shape[0]))
    event_mask = event.astype(np.float32) / 255.0
    event_mask_3ch = np.stack([event_mask] * 3, axis=-1)

    # Extract Laplacian edges from RGB
    lap = cv2.Laplacian(rgb, cv2.CV_64F)
    lap = np.clip(lap, 0, 255).astype(np.uint8)

    # Blend edges guided by event map
    fused = rgb + ALPHA * (lap * event_mask_3ch)
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    # Save result
    out_path = os.path.join(OUTPUT_DIR, rgb_name)
    cv2.imwrite(out_path, fused)
    print(f"✅ Fused: {rgb_name} → {out_path}")

print("🎉 All images fused successfully!")
