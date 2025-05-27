import os
import cv2
import numpy as np

"""
This script applies a simple yet effective method for enhancing blurry frames using event-guided edge sharpening. 
It computes a temporal average of three consecutive RGB frames (t-1, t, and t+1) and uses the event map at time t to 
identify regions likely affected by motion blur. 
A Laplacian edge map is computed from the averaged frame and selectively injected into the result using the normalized
event mask as a guide. 
The process improves perceived sharpness in regions with detected motion while leaving static areas untouched.
"""


# === Configuration ===
RGB_DIR = "extracted_frames_scene_2"           # Directory containing extracted RGB frames
EVENT_DIR = "event_maps"                       # Directory containing grayscale event maps
OUTPUT_DIR = "event_guided_deblurred_v2"       # Output directory for enhanced frames
ALPHA_BASE = 0.8                                # Base strength for edge injection
EVENT_THRESHOLD = 0.2                           # Threshold for event mask binarization
DEBUG = False                                   # Enable visual output for debugging

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load sorted file names
rgb_filenames = sorted([f for f in os.listdir(RGB_DIR) if f.endswith(".png")])
event_filenames = sorted([f for f in os.listdir(EVENT_DIR) if f.endswith(".png")])
assert len(rgb_filenames) == len(event_filenames), "Mismatch in number of RGB and event map files."


def read_image(path, grayscale=False):
    """Reads an image in color or grayscale."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)


for i in range(1, len(rgb_filenames) - 1):
    # Load three consecutive RGB frames (t-1, t, t+1)
    frame_prev = read_image(os.path.join(RGB_DIR, rgb_filenames[i - 1]))
    frame_curr = read_image(os.path.join(RGB_DIR, rgb_filenames[i]))
    frame_next = read_image(os.path.join(RGB_DIR, rgb_filenames[i + 1]))

    # Load and normalize event map for frame t
    event_map = read_image(os.path.join(EVENT_DIR, event_filenames[i]), grayscale=True)
    event_map = cv2.resize(event_map, (frame_curr.shape[1], frame_curr.shape[0]))
    event_map_normalized = event_map.astype(np.float32) / 255.0
    event_map_clipped = np.clip(event_map_normalized, 0, 1)

    # Compute pixel-wise blending weights
    weight_prev = 1.0 - event_map_clipped
    weight_curr = 0.5 * np.ones_like(event_map_clipped)
    weight_next = event_map_clipped

    # Normalize weights so they sum to 1 at each pixel
    weight_sum = weight_prev + weight_curr + weight_next + 1e-6  # small epsilon for numerical safety
    weight_prev /= weight_sum
    weight_curr /= weight_sum
    weight_next /= weight_sum

    # Expand to 3 channels for RGB frame fusion
    weight_prev_3ch = np.stack([weight_prev] * 3, axis=-1)
    weight_curr_3ch = np.stack([weight_curr] * 3, axis=-1)
    weight_next_3ch = np.stack([weight_next] * 3, axis=-1)

    # Compute event-aware temporal fusion of RGB frames
    fused_frame = (
        frame_prev.astype(np.float32) * weight_prev_3ch +
        frame_curr.astype(np.float32) * weight_curr_3ch +
        frame_next.astype(np.float32) * weight_next_3ch
    ).astype(np.uint8)

    # Compute Laplacian edge map of the fused frame
    laplacian = cv2.Laplacian(fused_frame, cv2.CV_64F)
    laplacian_normalized = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-6)
    laplacian_scaled = (laplacian_normalized * 255).astype(np.uint8)

    # Generate binary event mask for selective sharpening
    event_mask_binary = (event_map_clipped >= EVENT_THRESHOLD).astype(np.float32)
    event_mask_3ch = np.stack([event_mask_binary] * 3, axis=-1)

    # Dynamically adjust enhancement strength based on global motion level
    motion_strength = np.mean(event_map_clipped)
    alpha = ALPHA_BASE + motion_strength * 1.0

    # Inject Laplacian edges into fused RGB frame guided by the event mask
    enhanced_frame = fused_frame + alpha * (laplacian_scaled * event_mask_3ch)
    enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)

    # Save the enhanced output frame
    output_path = os.path.join(OUTPUT_DIR, rgb_filenames[i])
    cv2.imwrite(output_path, enhanced_frame)
    print(f"Enhanced: {rgb_filenames[i]} — Alpha used: {alpha:.2f}")

    # Optional debugging visualization
    if DEBUG:
        event_colored = cv2.cvtColor(event_map, cv2.COLOR_GRAY2BGR)
        debug_view = np.hstack([frame_curr, event_colored, fused_frame, enhanced_frame])
        cv2.imshow("Current | Event | Fused | Enhanced", debug_view)
        cv2.waitKey(0)

print("Processing complete. All frames enhanced and saved.")
