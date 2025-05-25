import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# === CONFIG ===
bag_path = Path("converted_scene_2")
csv_path = Path("dvs_events_fake_aligned.csv")
output_dir = Path("event_maps")
image_topic = '/dvs/image_raw'
event_window_sec = 0.2  # 100ms before each image

output_dir.mkdir(exist_ok=True)

# === LOAD EVENTS ===
print("📥 Loading event CSV...")
df = pd.read_csv(csv_path)

# === GET IMAGE TIMESTAMPS ===
print("🕐 Extracting image timestamps...")
image_timestamps = []

with AnyReader([bag_path]) as reader:
    connections = [x for x in reader.connections if x.topic == image_topic]
    for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
        image_timestamps.append((i, timestamp / 1e9))  # ns → sec

# === GET IMAGE SHAPE ===
print("📐 Reading sample image size...")
sample_img = cv2.imread('extracted_frames_scene_2/frame_0000.png')
height, width = sample_img.shape[:2]

# === GENERATE EVENT MAPS ===
print("🧠 Generating event maps...")
for i, image_ts in image_timestamps:
    start_ts = image_ts - event_window_sec
    df_window = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= image_ts)]
    print(f"[Frame {i}] Event count: {len(df_window)}")


    # Create signed event map
    event_map = np.zeros((height, width), dtype=np.int32)

    for _, row in df_window.iterrows():
        x, y, polarity = int(row['x']), int(row['y']), int(row['polarity'])
        if 0 <= x < width and 0 <= y < height:
            event_map[y, x] += 1 if polarity else -1

    # Normalize to 0–255
    abs_map = np.abs(event_map)
    norm_map = np.clip(abs_map * 5, 0, 255).astype(np.uint8)  # boost contrast 5x


    filename = output_dir / f"event_map_{i:04d}.png"
    cv2.imwrite(str(filename), norm_map)
    print(f"✅ Saved {filename.name}")

print("🎉 All event maps generated!")
