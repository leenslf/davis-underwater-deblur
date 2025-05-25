import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
RGB_DIR = "downsampled_rgb"
EVENT_MAP_DIR = "downsampled_event_maps"
OUTPUT_DIR = "downsampled_boosted_rgb"
PREVIEW_DIR = "preview_boosted_rgb"
ALPHA = 1.5   # Sharpening intensity multiplier
LAPLACIAN_SCALE = 2.0  # Multiply edge strength
PREVIEW_BOOST = 1.1   # Brighten preview image for visualization

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Get matching filenames
rgb_files = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
event_files = sorted([f for f in os.listdir(EVENT_MAP_DIR) if f.endswith(".png")])
assert len(rgb_files) == len(event_files), "Mismatch in number of frames and event maps"

for rgb_name, event_name in zip(rgb_files, event_files):
    # Load images
    rgb_path = os.path.join(RGB_DIR, rgb_name)
    event_path = os.path.join(EVENT_MAP_DIR, event_name)
    rgb = cv2.imread(rgb_path).astype(np.float32)
    event_map = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or event_map is None:
        print(f"⚠️ Failed to load {rgb_name} or {event_name}. Skipping.")
        continue

    # Resize event map to match RGB shape
    event_map = cv2.resize(event_map, (rgb.shape[1], rgb.shape[0]))
    event_mask = event_map.astype(np.float32) / 255.0
    event_mask_3ch = np.stack([event_mask] * 3, axis=-1)

    # Apply Gaussian blur before Laplacian
    blurred = cv2.GaussianBlur(rgb, (3, 3), 0)

    # Compute Laplacian (signed float32), boost it
    lap = cv2.Laplacian(blurred, cv2.CV_32F)
    lap *= LAPLACIAN_SCALE

    # Event-guided boosting
    boosted = rgb + ALPHA * (lap * event_mask_3ch)
    boosted_uint8 = np.clip(boosted, 0, 255).astype(np.uint8)

    # Save final boosted image
    cv2.imwrite(os.path.join(OUTPUT_DIR, rgb_name), boosted_uint8)

    # Save visual preview (brightened)
    boosted_preview = np.clip(boosted * PREVIEW_BOOST, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(PREVIEW_DIR, f"preview_{rgb_name}"), boosted_preview)

    # Save diff map for sanity check
    diff = cv2.absdiff(boosted_uint8, rgb.astype(np.uint8))
    cv2.imwrite(os.path.join(PREVIEW_DIR, f"diff_{rgb_name}"), diff)

    # # Optional histogram plot (for first image only)
    # if rgb_name == rgb_files[0]:
    #     plt.hist(boosted_uint8.ravel(), bins=256)
    #     plt.title("Pixel Intensity Histogram (Boosted)")
    #     plt.xlabel("Pixel Value")
    #     plt.ylabel("Frequency")
    #     plt.savefig(os.path.join(PREVIEW_DIR, "histogram_boosted.png"))
    #     plt.close()

    print(f"✅ {rgb_name} boosted and saved | Max pixel: {boosted_uint8.max()}")

print("🎉 All done! Check both the boosted and preview directories.")
