# Event-Guided Motion Deblurring for Underwater Images

This repository contains a lightweight image enhancement pipeline that uses event camera data to improve underwater image clarity. The goal is to restore sharper structure in blurry RGB frames using event-driven motion cues, without relying on deep learning.

---

## 👀 What We Did

1. Extracted RGB frames and event streams from underwater DAVIS recordings.
2. Generated per-frame **event maps** indicating motion edges.
3. Downsampled and aligned RGB and event data.
4. Developed an event-guided gradient fusion method to restore structure.

## What's Next?
Tuning the pipeline to reduce halo artifacts.

---

## 📦 Dataset

We used the **DAVIS-NUIUIED** dataset, released alongside the paper:

> **"RGB/Event signal fusion framework for multi-degraded underwater image enhancement"**  
> [ResearchGate](https://www.researchgate.net/publication/381024944_RGBEvent_signal_fusion_framework_for_multi-degraded_underwater_image_enhancement)

This public dataset includes:
- **Synchronized RGB and event data** from underwater scenes
- Collected using a DAVIS sensor (Dynamic and Active-pixel Vision Sensor)
- Packaged as ROS bag files containing:
  - `/dvs/image_raw`: standard RGB frames
  - `/dvs/events`: asynchronous event streams
  - `/dvs/imu`: inertial measurements (unused in this project)

We extracted RGB frames and generated grayscale event maps to guide motion-aware enhancement. The dataset is particularly challenging due to low visibility, motion blur, and illumination drop-off — making it an ideal testbed for event-based deblurring.


---

## 🚀 Pipeline Summary

We compute Laplacian edge gradients from event maps and inject them into the luminance channel of RGB images. This adds edge clarity where motion blur is most damaging.

Key features:

* **Adaptive contrast boosting** via CLAHE
* **Gradient clipping** to reduce halo artifacts
* **Edge-preserving filtering** using bilateral filters

---

## 🔢 Scripts

| Script                             | Description                                                |
| ---------------------------------- | ---------------------------------------------------------- |
| `extract_events.py`                | Extracts event arrays from `.bag` file                     |
| `dump_frames.py`                   | Extracts RGB frames from rosbag                            |
| `generate_event_maps.py`           | Converts event streams into grayscale event maps           |
| `downsample_event_rgb.py`          | Resizes RGB and event maps for consistent input shape      |
| `boost_with_events.py`             | Laplacian-based edge sharpening using event-weighted masks |
| `fuse_event_rgb.py`                | Initial RGB/event luminance fusion method                  |
| `blend_event_deblur.py`            | Blends event map with RGB luminance (simple enhancement)   |
| `blend_event_deblur_halo_fixed.py` | Final version with halo reduction + filtering              |
| `shift_event_timestamps.py`        | Aligns event timestamps with RGB frames                    |
| `get_first_image_timestamp.py`     | Helper script for data synchronization                     |

---

## 📂 Directory Structure

> Note: the following folders and files are **not pushed to GitHub** due to size.

* `*.bag` files — raw DAVIS recordings
* `downsampled_rgb/` — extracted and resized RGB images
* `event_maps/` — generated grayscale event maps
* `blended_deblurred_rgb/` — output images from the final pipeline

---

## 🌐 Project Info

* Course: **Computational Photography** @ Hacettepe University
* Author: **Leen Said**

