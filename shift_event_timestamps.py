import pandas as pd

first_image_ts = 1649246177.90659
original_event_start = 1640312682.95
offset = first_image_ts - original_event_start

print(f"📦 Shifting event timestamps by {offset:.2f} seconds")

df = pd.read_csv("dvs_events_scene_2.csv")
df['timestamp'] = df['timestamp'] + offset
df.to_csv("dvs_events_fake_aligned.csv", index=False)

print("✅ Saved: dvs_events_fake_aligned.csv")
