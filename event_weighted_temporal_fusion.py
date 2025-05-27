import os
import cv2
import numpy as np

"""
This script performs event-guided deblurring using adaptive temporal fusion. 
For each frame, it loads the previous, current, and next RGB frames and combines them using pixel-wise weights
derived from the event map. 
Regions with high event activity (indicating motion) are weighted more heavily toward the next frame, while 
static regions favor the previous frame. 
The result is a temporally smoothed frame that prioritizes sharper sources based on motion cues. 
A Laplacian edge map is computed from the fused RGB result and added back selectively in high-event regions to 
recover fine structures lost due to blur.
"""

# === Configuration ===
RGB_DIR = "extracted_frames_scene_2"            # Directory containing RGB frames
EVENT_DIR = "event_maps"                        # Directory containing grayscale event maps
OUTPUT_DIR = "event_guided_deblurred"           # Directory to save the output frames
ALPHA = 0.8                                      # Blending strength for edge enhancement

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and sort file lists
rgb_filenames = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
event_filenames = sorted([f for f in os.listdir(EVENT_DIR) if f.endswith(".png")])
assert len(rgb_filenames) == len(event_filenames), "Mismatch in RGB and event map file count."

def read_image(path, grayscale=False):
    """Reads an image in either grayscale or color."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)

# Process each frame, skipping the first and last for temporal averaging
for i in range(1, len(rgb_filenames) - 1):
    # Load three consecutive RGB frames: t-1, t, t+1
    frame_prev = read_image(os.path.join(RGB_DIR, rgb_filenames[i - 1]))
    frame_curr = read_image(os.path.join(RGB_DIR, rgb_filenames[i]))
    frame_next = read_image(os.path.join(RGB_DIR, rgb_filenames[i + 1]))

    # Compute temporal average RGB frame
    temporal_average = np.mean([frame_prev, frame_curr, frame_next], axis=0).astype(np.uint8)

    # Load and resize corresponding event map
    event_map = read_image(os.path.join(EVENT_DIR, event_filenames[i]), grayscale=True)
    event_map = cv2.resize(event_map, (temporal_average.shape[1], temporal_average.shape[0]))

    # Normalize event map to [0, 1] and expand to 3 channels
    event_mask = event_map.astype(np.float32) / 255.0
    event_mask_3ch = np.stack([event_mask] * 3, axis=-1)

    # Compute Laplacian edges of the temporally averaged RGB frame
    laplacian_edges = cv2.Laplacian(temporal_average, cv2.CV_64F)
    laplacian_edges = np.clip(laplacian_edges, 0, 255).astype(np.uint8)

    # Blend edge-enhanced features with the original using the event mask
    enhanced_frame = temporal_average + ALPHA * (laplacian_edges * event_mask_3ch)
    enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)

    # Save the enhanced result
    output_path = os.path.join(OUTPUT_DIR, rgb_filenames[i])
    cv2.imwrite(output_path, enhanced_frame)
    print(f"Enhanced: {rgb_filenames[i]}")

print("All frames processed and saved using event-guided deblurring.")
