import os
import cv2

# === Customize these paths ===
input_rgb_dir = "extracted_frames_scene_2"
input_event_dir = "event_maps"
output_rgb_dir = "downsampled_rgb"
output_event_dir = "downsampled_event_maps"

# Target size: set to what your model prefers (width, height)
target_size = (256, 192)   # AOD-Net default is 640x480

# Create output folders
os.makedirs(output_rgb_dir, exist_ok=True)
os.makedirs(output_event_dir, exist_ok=True)

# Get file lists
rgb_files = sorted([f for f in os.listdir(input_rgb_dir) if f.endswith(".png")])
event_files = sorted([f for f in os.listdir(input_event_dir) if f.endswith(".png")])

for rgb_file, event_file in zip(rgb_files, event_files):
    rgb_path = os.path.join(input_rgb_dir, rgb_file)
    event_path = os.path.join(input_event_dir, event_file)

    rgb_img = cv2.imread(rgb_path)
    event_img = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)

    orig_h, orig_w = rgb_img.shape[:2]
    new_w, new_h = target_size

    # Check downsampling status
    is_downsample = orig_w > new_w or orig_h > new_h
    status = "⬇️ DOWNSAMPLING" if is_downsample else "⚠️ NOT downsampling"

    print(f"{rgb_file}: {orig_w}x{orig_h} → {new_w}x{new_h} → {status}")

    # Resize and save
    rgb_resized = cv2.resize(rgb_img, target_size, interpolation=cv2.INTER_AREA)
    event_resized = cv2.resize(event_img, target_size, interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(output_rgb_dir, rgb_file), rgb_resized)
    cv2.imwrite(os.path.join(output_event_dir, event_file), event_resized)

print("✅ Downsampling check + resizing complete.")
