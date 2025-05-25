import os
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

bag_path = Path("converted_scene_2")  
output_folder = "extracted_frames_scene_2"
os.makedirs(output_folder, exist_ok=True)

# Topic we want to extract frames from
image_topic = '/dvs/image_raw'

# Open the bag
with AnyReader([bag_path]) as reader:
    # Get connections for the image topic
    connections = [x for x in reader.connections if x.topic == image_topic]

    if not connections:
        print(f"No messages found for topic {image_topic}")
        exit(1)

    # Loop through messages and save them as PNGs
    for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
        msg = reader.deserialize(rawdata, conn.msgtype)

        # Get height, width, and encoding
        height = msg.height
        width = msg.width
        encoding = msg.encoding

        # Always uint8 for typical images
        dtype = np.uint8

        # Decode based on encoding
        if encoding in ['rgb8', 'bgr8']:
            img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width, 3))
        elif encoding in ['mono8', '8UC1']:
            img = np.frombuffer(msg.data, dtype=dtype).reshape((height, width))
        else:
            print(f"Unsupported encoding: {encoding}")
            continue  # Skip unknown encodings

        # Save the image
        filename = f"{output_folder}/frame_{i:04d}.png"
        cv2.imwrite(filename, img)

        print(f"✅ Saved {filename}")
