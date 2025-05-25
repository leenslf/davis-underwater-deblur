import os
import csv
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

# Path to converted bag folder
bag_path = Path("converted_scene_2")  # ⬅️ updated
output_csv = "dvs_events_scene_2.csv"
event_topic = "/dvs/events"


# Open bag file
with AnyReader([bag_path]) as reader:
    connections = [x for x in reader.connections if x.topic == event_topic]

    if not connections:
        print(f"No messages found for topic {event_topic}")
        exit(1)

    print(f"Extracting events to {output_csv}...")

    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'timestamp', 'polarity'])  # header

        for i, (conn, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, conn.msgtype)

            for e in msg.events:
                writer.writerow([e.x, e.y, e.ts.sec + e.ts.nanosec * 1e-9, int(e.polarity)])

    print("✅ Done!")

