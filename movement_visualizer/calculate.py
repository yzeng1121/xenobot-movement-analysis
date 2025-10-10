# 
#   calculate.py
#   the purpose of this file is to calculate the linear speed, heading, & angular
#   speed for each xenobot from tracking data taken from trackR
#

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# input: data frame holding tracking data
def calculate_metrics(df):
    df = df.sort_values(['track_fixed', 'frame']).copy()

    df['linear_speed'] = np.nan
    df['heading'] = np.nan
    df['angular_speed'] = np.nan

    # group by xenobot
    for xenobot_id, group in df.groupby('track_fixed'):
        indexes = group.index # for this specific group 

        for i in range(1, len(group)):
            curr_i = indices[i]
            prev_i = indices[i - 1]

            curr_x = group.loc[curr_i, 'x']
            prev_x = group.loc[prev_i, 'x']
            curr_y = group.loc[curr_i, 'y']
            prev_y = group.loc[prev_i, 'y']

            curr_frame = group.loc[curr_i, 'frame']
            prev_frames = group.loc[prev_i, 'frame']
            time_diff = curr_frame - prev_frames

            if time_diff <= 0: 
                continue

            # calculate the linear speed . . . 
            dist = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            linear_speed = dist / time_diff
            df.loc[curr_i, 'linear_speed'] = linear_speed

            # calculate heading (angle of movement) . . .
            dx = curr_x - prev_x
            dy = curr_y - prev_y
            heading = np.arctan2(dy, dx)
            df.loc[curr_i, 'heading'] = heading

            # calculate angular speed (magnitude of change in angles) . . .
            if i >= 2:
                prev_prev_i = indexes[i - 2]

                prev_prev_x = group.loc(prev_prev_i, 'x')
                prev_prev_y = group.loc(prev_prev_i, 'y')
                prev_dx = prev_x - prev_prev_x
                prev_dy = prev_y - prev_prev_y
                prev_heading = np.arctan2(prev_dy, prev_dx)

                angle_diff = heading - prev_heading

                # normalize angle to [-pi, pi]
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

                prev_prev_frame = group.loc[prev_prev_i, 'frame']
                angular_time_diff = curr_frame - prev_prev_frame

                if (angular_time_diff > 0):
                    angular_speed = angle_diff / angular_time_diff 
                    df.loc[curr_i, 'angular_speed'] = angular_speed
            
    return df

def main(): 
    if len(sys.argv) != 2:
        print("Usage: python calculate.py <input_csv_path")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    # error check
    if not input_path.exists():
        print(f"Error: file '{input_path} not found")
        sys.exit(1)
    
    # open + read csv file 
    try:
        df = pd.read_csv(input_path)
        print(f"loaded {len(df)} rows from {input_path.name}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    required_cols = ['frame', 'x', 'y', 'track_fixed']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Error: missing required columns: {missing_cols}")
        sys.exit(1) 

    df_with_metrics = calculate_movement_metrics(df)
    output_path = input_path.parent / f"{input_path.stem}_processed.csv"

    # save data to new CSV file
    df_with_metrics.to_csv(output_path, index = False)

    return df_with_metrics

if __name__ == "__main__":
    main()


    

        
    




            


