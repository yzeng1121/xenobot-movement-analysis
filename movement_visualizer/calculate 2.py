import pandas as pd
import numpy as np
from pathlib import Path

# take in user query
file_path = input("Please paste the path to the CSV file: ")

# load in the csv file 
csv = pd.read_csv(file_path)

# sort data in csv based on time
csv = csv.sort_values(by=["track_fixed", "frame"]).reset_index(drop=True)

# func to normalize the angle inputs
def norm_angles(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# init new columns to non-numbers in csv
csv["linear_speed"] = np.nan
csv["heading"] = np.nan
csv["angular_speed"] = np.nan

# group by tracks
for track_id, group in csv.groupby("track_fixed"):
    i = group.index

    # points for calculation
    x = group["x"].values
    y = group["y"].values
    t = group["frame"].values

    # store differences b/w consecutive values
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t).astype(float)

    # prevents zero division error
    dt[dt == 0] = np.nan

    # linear speed 
    speed = np.sqrt(dx ** 2 + dy ** 2) / dt

    # heading (direction) in radians
    heading = np.arctan2(dy, dx)

    # angular speed (change in direction over time)
    dtheta = np.diff(heading)
    dtheta = np.array([norm_angles(val) for val in dtheta])
    a_speed = dtheta / dt[1:]

    # populate csv
    csv.loc[i[1:], "linear_speed"] = speed
    csv.loc[i[1:], "heading"] = heading
    csv.loc[i[2:], "angular_speed"] = a_speed

# save result
path = Path(file_path)
newName = path.stem
output_p = path.parent / (newName + "_processed.csv")

csv.to_csv(output_p, index = False)
print("file has been processed + saved to " + str(output_p) + "!" )