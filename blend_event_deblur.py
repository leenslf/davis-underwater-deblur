# import os
# import cv2
# import numpy as np

# # === CONFIG ===
# RGB_DIR = "downsampled_rgb"
# EVENT_MAP_DIR = "downsampled_event_maps"
# OUTPUT_DIR = "blended_deblurred_rgb"
# ALPHA = 0.4  # blend ratio between original luminance and event map
# MIN_EVENT_THRESHOLD = 15  # skip frames with max event pixel below this

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
# event_files = sorted([f for f in os.listdir(EVENT_MAP_DIR) if f.endswith(".png")])
# assert len(rgb_files) == len(event_files), "Mismatch in number of frames and event maps"

# # CLAHE setup (for adaptive histogram equalization)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# for rgb_file, event_file in zip(rgb_files, event_files):
#     rgb_path = os.path.join(RGB_DIR, rgb_file)
#     event_path = os.path.join(EVENT_MAP_DIR, event_file)

#     rgb = cv2.imread(rgb_path)
#     event = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

#     if rgb is None or event is None:
#         print(f"⚠️ Skipping {rgb_file} or {event_file}")
#         continue

#     event = cv2.resize(event, (rgb.shape[1], rgb.shape[0]))
    
#     # Check if event map is too weak
#     if event.max() < MIN_EVENT_THRESHOLD:
#         print(f"⛔ Skipping {rgb_file} — event map too faint (max={event.max()})")
#         continue

#     # Enhance contrast with CLAHE
#     event_eq = clahe.apply(event)

#     # Boost event edge brightness
#     event_boosted = cv2.convertScaleAbs(event_eq, alpha=2.0, beta=10)
#     event_denoised = cv2.medianBlur(event_boosted, 3)

#     # Convert RGB to YUV, blend luminance
#     yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
#     y_original = yuv[:, :, 0].astype(np.float32)
#     y_event = event_denoised.astype(np.float32)

#     y_blend = np.clip((1 - ALPHA) * y_original + ALPHA * y_event, 0, 255).astype(np.uint8)

#     yuv[:, :, 0] = y_blend
#     fused = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

#     out_path = os.path.join(OUTPUT_DIR, rgb_file)
#     cv2.imwrite(out_path, fused)

#     print(f"✅ Saved: {out_path} | event max: {event.max()}")

# print("🎉 Done! All usable frames processed with event-guided luminance fusion.")


import os
import cv2
import numpy as np

# === CONFIG ===
RGB_DIR = "downsampled_rgb"
EVENT_MAP_DIR = "downsampled_event_maps"
OUTPUT_DIR = "blended_deblurred_rgb"
ALPHA = 0.6                     # Weight for adding edge gradients
MIN_EVENT_THRESHOLD = 15        # Skip if event map is too weak
LAPLACIAN_CLIP = 30             # Clamp Laplacian gradient to avoid artifacts
ENABLE_CONTRAST_STRETCH = True
ENABLE_GAMMA_CORRECTION = True
GAMMA = 1.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
event_files = sorted([f for f in os.listdir(EVENT_MAP_DIR) if f.endswith(".png")])
assert len(rgb_files) == len(event_files), "Mismatch in number of frames and event maps"

# CLAHE setup for adaptive contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for rgb_file, event_file in zip(rgb_files, event_files):
    rgb_path = os.path.join(RGB_DIR, rgb_file)
    event_path = os.path.join(EVENT_MAP_DIR, event_file)

    rgb = cv2.imread(rgb_path)
    event = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or event is None:
        print(f"⚠️ Skipping {rgb_file} or {event_file}")
        continue

    # Resize event map to match image size
    event = cv2.resize(event, (rgb.shape[1], rgb.shape[0]))

    # Skip if too faint
    if event.max() < MIN_EVENT_THRESHOLD:
        print(f"⛔ Skipping {rgb_file} — weak event map (max={event.max()})")
        continue

    # Step 1: Enhance the event map with CLAHE and denoise
    event_eq = clahe.apply(event)
    event_denoised = cv2.medianBlur(event_eq, 3)

    # Step 2: Convert RGB to YUV and extract original luminance
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YUV)
    y_original = yuv[:, :, 0].astype(np.float32)

    # Step 3: Extract edge structure from the event map
    event_edges = cv2.Laplacian(event_denoised, cv2.CV_32F)
    event_edges = np.clip(event_edges, -LAPLACIAN_CLIP, LAPLACIAN_CLIP)

    # Step 4: Add edge gradients to original luminance
    y_boosted = np.clip(y_original + ALPHA * event_edges, 0, 255).astype(np.uint8)

    # Step 5: Replace Y and convert back to RGB
    yuv[:, :, 0] = y_boosted
    fused = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # Step 6: Optional contrast stretch
    # if ENABLE_CONTRAST_STRETCH:
    #     fused = cv2.convertScaleAbs(fused, alpha=1.1, beta=5)

    # Step 7: Optional gamma correction
    if ENABLE_GAMMA_CORRECTION:
        lut = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)]).astype("uint8")
        fused = cv2.LUT(fused, lut)

    # Save result
    out_path = os.path.join(OUTPUT_DIR, rgb_file)
    cv2.imwrite(out_path, fused)

    print(f"✅ {rgb_file} processed | event max: {event.max()}")

print("🎉 All frames processed with event-guided edge boosting.")
