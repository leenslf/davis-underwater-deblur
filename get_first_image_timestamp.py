from pathlib import Path
from rosbags.highlevel import AnyReader

bag_path = Path("converted_scene_2")
image_topic = "/dvs/image_raw"

with AnyReader([bag_path]) as reader:
    conns = [x for x in reader.connections if x.topic == image_topic]
    for _, ts, _ in reader.messages(connections=conns):
        first_image_ts = ts / 1e9  # convert from ns to sec
        break

print("✅ First RGB timestamp:", first_image_ts)
