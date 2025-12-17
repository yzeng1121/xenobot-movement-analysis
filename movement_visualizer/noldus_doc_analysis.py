import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
import seaborn as sns

from scipy.stats import circvar
from collections import Counter

import os
os.environ['R_HOME'] = '/opt/homebrew/Cellar/r/4.5.1/lib/R'

import warnings
warnings.filterwarnings('ignore')

try:
    import rpy2.robjects as ro
    from rpy2.robjects import r, StrVector, IntVector, FloatVector
    from rpy2.robjects.packages import importr
except ImportError as e:
    raise ImportError("error: R packages failed to be installed") from e

try:
    r_cec = importr('CEC')
except Exception as e:
    raise ImportError("error: R packages failed to be installed") from e


#  name:      load_data
#  purpose:   takes in csv file and stores it into a Python dataframe/array allowing the program
#             to manipulate and analyze
#  arguments: file path of Noldus csv file in user's computer
#  returns:   the dataframe populated with input
#  effects:   converts angular and linear speed units to per sec rather than per min
def load_data(file_path):
    if file_path.endswith('.csv'):
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, header=33, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"couldn't read CSV file w/ any supported encoding: {file_path}")
            
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path, header=33)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    df.columns = [
        'trial_time', 
        'recording_time', 
        'x_center', 
        'y_center', 
        'area', 
        'area_change', 
        'elongation', 
        'distance_moved', 
        'velocity', 
        'heading', 
        'angular_velocity',
        'result_1'
    ]

    int_cols = [
        'trial_time', 
        'recording_time', 
        'x_center', 
        'y_center',  
        'velocity', 
        'heading', 
        'angular_velocity'
    ]

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors = 'coerce')

    df = df.dropna(subset = ['velocity', 'heading', 'angular_velocity'])

    df = df.rename(columns = {
        'velocity': 'linear_speed',
        'angular_velocity': 'angular_speed',
        'trial_time': 'time'
    })

    df = convert_velocity_units(df)

    return df


#  name:      convert velocity units
#  purpose:   converts per min to per sec (angular and linear velocity)
#  arguments: populated data frame
#  returns:   the data frame with fixed velocity units
#  effects:   converts angular and linear speed units to per sec rather than per min
def convert_velocity_units(df):
    if 'linear_speed' in df.columns:
        df['linear_speed'] = df['linear_speed'] / 60.0

    if 'angular_speed' in df.columns:
        df['angular_speed'] = df['angular_speed'] / 60.0

    return df


#  name:      break into chunks
#  purpose:   breaks the data frame into 30 second chunks at 30fps
#  arguments: data frame, duration of chunks (30 seconds matches the research paper), frames per 
#             second
#  returns:   array of time chunks
#  effects:   none
def break_into_chunks(df, chunk_duration_secs = 30, fps = 30):
    chunks = []
    chunk_duration_frames = chunk_duration_secs * fps

    df = df.sort_values('time').reset_index(drop = True)

    if 'frame' not in df.columns:
        df['frame'] = (df['time'] * fps).astype(int)

    start_frame = df['frame'].min()
    end_frame = df['frame'].max()

    for chunk_start in range(start_frame, end_frame, chunk_duration_frames):
        chunk_end = chunk_start + chunk_duration_frames
        chunk_data = df[(df['frame'] >= chunk_start) & (df['frame'] < chunk_end)].copy()

        if len(chunk_data) > 10:
            chunks.append({
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'data': chunk_data
            })
    
    return chunks


#  name:      calculate metrics
#  purpose:   calculates the straightness index as 1 minus the circular variance of the headings
#             during the block and the gyration index as 1 minus the circular variance of the and
#             averages to allow for relative comparison
#             angular speeds during the block divided by the circular variance of the angular speeds
#  arguments: data in chunks, reference circular variance i.e. circular variance of the angular 
#             speeds
#  returns:   array of the calculated metrics
#  effects:   none
def calculate_metrics(chunk, ref_ang_circvar):
    data = chunk['data']

    if len(data) < 2:
        return None
    
    straightness = 1 - circvar(np.radians(data['heading'].dropna()))

    if not data['angular_speed'].isna().all() and ref_ang_circvar > 0:
        chunk_circvar = circvar(np.radians(data['angular_speed'].dropna()))
        gyration = 1 - (chunk_circvar / ref_ang_circvar)
    else:
        gyration = 0

    features = {
        'chunk_start': chunk['chunk_start'],
        'mean_speed': data['linear_speed'].mean(),
        'mean_angular_speed': data['angular_speed'].mean(),
        'straightness': straightness,
        'gyration': gyration
    }

    return features


#  name:      compute ref angular circvar
#  purpose:   calculates the reference angular circular variance of angular speeds
#  arguments: populated data frame
#  returns:   reference angular circular variance of angular speeds
#  effects:   none
def compute_ref_angular_circvar(df):
    ang = np.radians(df['angular_speed'].dropna().values)
    if len(ang) < 2:
        return np.nan
    return circvar(ang)


#  name:      fit cec clustering
#  purpose:   uses R's CEC package to separate the trajectory chunks into categories of similar 
#             state behaviors
#  arguments: data frame holding data about each time chunk, the number of states in which case will 
#             always be 4 (idle, circling, linear, intermediate), the covariance type for the CEC 
#             package, which will be the default (spherical), the number of iterations to run the 
#             algorithm, arbitrarily choose to be 25, and the random seed set default to be 42 to 
#             allow for replicability
#  returns:   data frame of all clusters taken from the CEC algorithm
#  effects:   none
def fit_cec_clustering(
        features_df, 
        n_states = 4,
        cov_type = 'spherical',
        iterations = 25,
        random_seed = 42
):
    features = ['straightness', 'gyration']

    x = features_df[features].copy()
    x = x.fillna(0).values

    # handles no variance .. would break clustering
    if np.std(x[:, 0]) < 1e-10 or np.std(x[:, 1]) < 1e-10:
        x = x + np.random.normal(0, 1e-6, x.shape)

    unique_x, unique_indices = np.unique(x, axis = 0, return_inverse = True)

    if len(unique_x) < len(x):
        x_for_clustering = unique_x
    else:
        x_for_clustering = x

    if len(x_for_clustering) < n_states:
        raise ValueError(f"error: not enough data points for {n_states} clusters")

    np.random.seed(random_seed)
    r('set.seed({})'.format(random_seed))

    card_min_value = max(1, int(0.05 * len(x_for_clustering)))

    try:
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter

        with localconverter(ro.default_converter + numpy2ri.converter):
            r_data = ro.r.matrix(
                FloatVector(x_for_clustering.T.flatten()),
                nrow = x_for_clustering.shape[0],
                ncol = x_for_clustering.shape[1]
            )

            try:
                result = r['cec'](
                    x = r_data,
                    centers = IntVector([n_states])[0],
                    type = StrVector([cov_type])[0],
                    iter_max = IntVector([iterations])[0],
                    card_min = IntVector([card_min_value])
                )
            except Exception as e:
                print(f"error: cec failed using spherical covariance: {e}")
                result = r['cec'](
                    x = r_data,
                    centers = IntVector([n_states])[0],
                    type = StrVector(['diagonal'])[0],
                    iter_max = IntVector([iterations])[0],
                    card_min = IntVector([card_min_value])
                )
        result_names = list(result.names())
        cluster_key = next((k for k in ('clusters', 'cluster', 'cluster.indices', 'cluster_idx')
                            if k in result_names), None)

        if cluster_key is None:
            raise RuntimeError("couldn't find cluster assignments in cec")
        
        clusters_raw = np.array(result[result_names.index(cluster_key)])

        if len(unique_x) < len(x):
            clusters_raw = clusters_raw[unique_indices]

        unique_labels = np.unique(clusters_raw)
        label_map = {old: i for i, old in enumerate(sorted(unique_labels))}
        clusters = np.array([label_map[l] for l in clusters_raw])

        return clusters
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"cec clustering failed: {e}")


#  name:      label states
#  purpose:   classifies each time chunk as one of the four states and adds a new row to the data
#             frame
#  arguments: data frame holding pre-existing time chunk data
#  returns:   revised data frame
#  effects:   none
def label_states(features_df):
    speed_25 = features_df['mean_speed'].quantile(0.25)
    straight_25 = features_df['straightness'].quantile(0.25)
    straight_75 = features_df['straightness'].quantile(0.75)
    
    labels = []
    
    for idx, row in features_df.iterrows():
        speed = row['mean_speed']
        straightness = row['straightness']
        
        if speed < speed_25: # low speed (bottom 25%)
            labels.append('idle')
        elif straightness > straight_75:
            labels.append('linear') # high straightness AND NOT idle
        elif straightness < straight_25: # low straightness AND NOT idle  
            labels.append('circling')
        else: # everything else
            labels.append('intermediate')
    
    return labels


#  name:      calculate transition probabilities
#  purpose:   calculates the transition probabilities between states according to data from chunks
#  arguments: data frame holding pre-existing time chunk data
#  returns:   array holding probabilities of transition states
#  effects:   none
def calculate_transition_probabilities(features_df):
    features_df = features_df.sort_values('chunk_start').reset_index(drop = True)

    transitions = []

    for i in range(len(features_df) - 1):
        curr_state = features_df.iloc[i]['state_label']
        next_state = features_df.iloc[i + 1]['state_label']
        transitions.append((curr_state, next_state))

    transition_counts = Counter(transitions)
    states = sorted(features_df['state_label'].unique())
    transition_matrix = pd.DataFrame(0.0, index = states, columns = states)

    for from_state in states:
        total_from = sum(count for (s1, s2), count in transition_counts.items() 
                          if s1 == from_state)
        
        if total_from > 0:
            for to_state in states:
                count = transition_counts.get((from_state, to_state), 0)
                transition_matrix.loc[from_state, to_state] = count / total_from

    return transition_matrix


#  name:      plot transition heatmap
#  purpose:   plots state transition probability matrix on user's screen
#  arguments: array holding probabilities of transition states, title of the output state 
#             transition probability matrix
#  returns:   none
#  effects:   none
def plot_transition_heatmap(transition_matrix, title = "State Transition Probabilities"):
    plt.figure(figsize = (8, 6))
    sns.heatmap(
        transition_matrix,
        annot = True, 
        fmt = '0.2f',
        cmap = 'Blues',
        vmin = 0,
        vmax = 1,
        cbar_kws = {'label': 'Transition Probability'}
    )
    plt.title(title, fontsize = 14, fontweight = 'bold')
    plt.xlabel('To State', fontsize = 12)
    plt.ylabel('From State', fontsize = 12)
    plt.tight_layout()

    plt.show()
    plt.close()


#  name:      plot clustering scatter
#  purpose:   plots state clustering scatterplot on user's screen
#  arguments: array holding probabilities of transition states, title of the output scatterplot
#  returns:   none
#  effects:   none
def plot_clustering_scatter(features_df, title = "Movement State Clustering"):
    fig, ax = plt.subplots(figsize = (10, 8))

    state_colors = {
        'idle': '#666666',
        'linear': '#FFD700',
        'circling': '#9370DB',
        'intermediate': '#FFA07A'
    }

    for state_label in features_df['state_label'].unique():
        state_data = features_df[features_df['state_label'] == state_label]
        ax.scatter(
            state_data['straightness'], state_data['gyration'],
            c = state_colors.get(state_label, '#000000'),
            label = state_label.capitalize(),
            s = 100,
            alpha = 0.6,
            edgecolors = 'black',
            linewidth = 0.5
        )

    ax.set_xlabel('Straightness Index', fontsize = 12, fontweight = 'bold')
    ax.set_ylabel('Gyration Index', fontsize = 12, fontweight = 'bold')
    ax.set_title(title, fontsize = 14, fontweight = 'bold')
    ax.legend(loc = 'best', fontsize = 10)
    ax.grid(True, alpha = 0.3, linestyle = '--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    plt.show()
    plt.close()


#  name:      plot markov chain
#  purpose:   plots markov chain on user's screen
#  arguments: array holding probabilities of transition states, title of the output markov chain
#  returns:   none
#  effects:   none
def plot_markov_chain(transition_matrix, title = "Markov Chain State Transitions"):
    fig, ax = plt.subplots(figsize = (12, 10))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    states = list(transition_matrix.index)
    n_states = len(states)

    positions = {}
    radius = 3.5
    center_x, center_y = 5, 5

    for i, state in enumerate(states):
        angle = 2 * np.pi * i / n_states - np.pi / 2
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        positions[state] = (x, y)

    state_colors = {
        'idle': '#666666',
        'linear': '#FFD700',
        'circling': '#9370DB',
        'intermediate': '#FFA07A'
    }

    circles = {}
    for state, (x, y) in positions.items():
        circle = Circle(
            (x, y), 
            0.5, 
            color=state_colors.get(state, '#CCCCCC'),
            ec='black', 
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
        circles[state] = circle
        
        ax.text(
            x, y, 
            state.capitalize(), 
            ha = 'center', 
            va = 'center',
            fontsize = 11, 
            fontweight = 'bold',
            zorder = 11
        )

    min_prob = 0.05

    for from_state in states:
        for to_state in states:
            prob = transition_matrix.loc[from_state, to_state]
            
            if prob < min_prob:
                continue
            
            x1, y1 = positions[from_state]
            x2, y2 = positions[to_state]
            
            if from_state == to_state:
                # Self-loop
                angle = 2 * np.pi * states.index(from_state) / n_states - np.pi / 2
                loop_x = x1 + 0.8 * np.cos(angle)
                loop_y = y1 + 0.8 * np.sin(angle)
                
                circle_loop = Circle(
                    (loop_x, loop_y),
                    0.3,
                    fill = False,
                    ec = 'black',
                    linewidth = 2 * prob,
                    linestyle = '-',
                    zorder = 5
                )
                ax.add_patch(circle_loop)
                
                # Add probability label
                label_x = loop_x + 0.4 * np.cos(angle)
                label_y = loop_y + 0.4 * np.sin(angle)
                ax.text(
                    label_x, label_y,
                    f'{prob:.2f}',
                    fontsize = 9,
                    bbox = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'),
                    zorder = 12
                )
            else:
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                
                offset = 0.5
                start_x = x1 + offset * dx / dist
                start_y = y1 + offset * dy / dist
                end_x = x2 - offset * dx / dist
                end_y = y2 - offset * dy / dist
                
                arrow = FancyArrowPatch(
                    (start_x, start_y),
                    (end_x, end_y),
                    arrowstyle = '->,head_width=0.4,head_length=0.8',
                    connectionstyle = f"arc3,rad=0.2",
                    color = 'black',
                    linewidth = 2 * prob,
                    zorder = 5,
                    alpha = 0.7
                )
                ax.add_patch(arrow)
                
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
       
                perp_dx = -dy / dist * 0.3
                perp_dy = dx / dist * 0.3
                
                ax.text(
                    mid_x + perp_dx, 
                    mid_y + perp_dy,
                    f'{prob:.2f}',
                    fontsize = 9,
                    bbox = dict(boxstyle = 'round,pad=0.3', facecolor = 'white', edgecolor = 'gray'),
                    zorder = 12,
                    ha = 'center'
                )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad = 20)
    
    legend_elements = [
        mpatches.Patch(color=state_colors.get(s, '#CCCCCC'), label = s.capitalize())
        for s in states
    ]
    ax.legend(
        handles = legend_elements,
        loc = 'upper right',
        fontsize = 10,
        title = 'States',
        title_fontsize = 11
    )
    
    ax.text(
        0.5, 0.3,
        'Arrow thickness = transition probability',
        fontsize = 10,
        style = 'italic',
        bbox = dict(boxstyle = 'round,pad=0.5', facecolor = 'lightyellow', alpha = 0.8)
    )
    
    plt.tight_layout()
    
    plt.show()
    plt.close()


#  name:      analyze single xenobot
#  purpose:   converts a single xenobot's trajectory data into chunks then clusters, then into  
#             movement states and then plots the features for an individual xenobot
#  arguments: path to file, number of states, duration of a single chunk, fps, covariance type for
#             cec algorithm, random seed arbitrarly at 42 to allow replication
#  returns:   array holding probabilities of transition states, state transition probability matrix 
#  effects:   none
def analyze_single_xenobot(
        file_path, 
        n_states = 4, 
        chunk_duration = 30, 
        fps = 30, 
        cov_type = 'spherical', 
        random_seed = 42
    ):
 
    df = load_data(file_path)
    chunks = break_into_chunks(df, chunk_duration_secs = chunk_duration, fps = fps)
    
    ref_ang_circvar = compute_ref_angular_circvar(df)
    
    features = [
        calculate_metrics(chunk, ref_ang_circvar)
        for chunk in chunks
    ]
    features = [f for f in features if f is not None]
    features_df = pd.DataFrame(features)

    clusters = fit_cec_clustering(
        features_df, 
        n_states = n_states, 
        cov_type = cov_type, 
        random_seed = random_seed
    )
    
    features_df['state'] = clusters
    features_df['state_label'] = label_states(features_df)

    transition_matrix = calculate_transition_probabilities(features_df)
    
    return features_df, transition_matrix


#  name:      analyze combined xenobot
#  purpose:   converts multiple xenobot's trajectory data into chunks then clusters, then into  
#             movement states and then plots the features for all xenobots
#  arguments: path to file, number of states, duration of a single chunk, fps, covariance type for
#             cec algorithm, random seed arbitrarly at 42 to allow replication
#  returns:   array holding probabilities of transition states, state transition probability matrix 
#  effects:   none
def analyze_combined_xenobots(
        file_paths, 
        n_states = 4, 
        chunk_duration = 30, 
        fps = 30, 
        cov_type = 'spherical', 
        random_seed = 42
    ):
    
    all_features = []

    all_dfs = [load_data(fp) for fp in file_paths]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    ref_ang_circvar = compute_ref_angular_circvar(combined_df)
    
    for i, file_path in enumerate(file_paths, 1):
        # load data
        df = load_data(file_path)
        df['file_id'] = f"xenobot_{i}" # identification

        chunks = break_into_chunks(df, chunk_duration_secs=chunk_duration, fps=fps)
        
        features = [
            calculate_metrics(chunk, ref_ang_circvar)
            for chunk in chunks
        ]
        features = [f for f in features if f is not None]

        for feature_dict in features:
            feature_dict['file_id'] = f"xenobot_{i}"
            feature_dict['xenobot_id'] = i
        
        all_features.extend(features)
        
    
    combined_features_df = pd.DataFrame(all_features)
    print(f"\total chunks across all xenobots: {len(combined_features_df)}")
    print(f"overall mean speed: {combined_features_df['mean_speed'].mean():.2f} mm/s")
    print(f"overall mean straightness: {combined_features_df['straightness'].mean():.3f}")
    print(f"overall mean gyration: {combined_features_df['gyration'].mean():.3f}")
    
    clusters = fit_cec_clustering(
        combined_features_df, 
        n_states = n_states, 
        cov_type = cov_type, 
        random_seed = random_seed
    )
    
    combined_features_df['state'] = clusters
    combined_features_df['state_label'] = label_states(combined_features_df)
 
    for state_label, count in combined_features_df['state_label'].value_counts().items():
        percentage = (count / len(combined_features_df)) * 100
        print(f"  {state_label.capitalize()}: {count} chunks ({percentage:.1f}%)")

    transition_matrices = []
    weights = []
    
    for xenobot_id in sorted(combined_features_df['xenobot_id'].unique()):
        xenobot_data = combined_features_df[combined_features_df['xenobot_id'] == xenobot_id].copy()
        
        if len(xenobot_data) > 1: 
            tm = calculate_transition_probabilities(xenobot_data)
            transition_matrices.append(tm)
            weights.append(len(xenobot_data))
    
    all_states = sorted(combined_features_df['state_label'].unique())
    
    normalized_matrices = []
    for tm in transition_matrices:
        tm_normalized = tm.reindex(index = all_states, columns = all_states, fill_value = 0)
        normalized_matrices.append(tm_normalized)
    
    combined_transition_matrix = sum(tm * w for tm, w in zip(normalized_matrices, weights)) / sum(weights)
    
    return combined_features_df, combined_transition_matrix



#  name:      main
#  purpose:   driver function
#  arguments: none
#  returns:   none
#  effects:   none
def main():
    # TODO: input the file paths of your CSV files here
    # NOTE: program assumes that each CSV file includes a single xenobot's data
    file_paths = [
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_1.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_2.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_3.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_4.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_5.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_6.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_7.csv',
        '/Users/yuxin/Desktop/xenobots/movement_visualizer/sample_data/sample_mixed_8.csv'
    ]
    

    import glob
    features_df, transition_matrix = analyze_combined_xenobots(
        file_paths,
        n_states = 4,
        chunk_duration = 5,
        fps = 30,
        cov_type='spherical',
        random_seed = 42 
    )
    
    print("\nGenerating visualizations...")
    
    plot_clustering_scatter(
        features_df,
        title = "Combined Xenobot Movement State Classification (CEC)\n(Straightness vs Gyration Index)"
    )
    
    plot_transition_heatmap(
        transition_matrix,
        title = "Combined State Transition Probability Matrix (CEC)"
    )

    plot_markov_chain(
        transition_matrix,
        title = "Markov Chain State Transitions (CEC)"
    )
    
    print("\nSaving results...")
    output_dir = '/Users/yuxin/Downloads/ethovision_csv_files/' # TODO: change to specific local dir
    
    features_df.to_csv(f'{output_dir}combined_xenobot_features_CEC.csv', index = False)
    transition_matrix.to_csv(f'{output_dir}combined_transition_matrix_CEC.csv')
    
    for xenobot_id in sorted(features_df['xenobot_id'].unique()):
        xenobot_data = features_df[features_df['xenobot_id'] == xenobot_id]
        xenobot_data.to_csv(f'{output_dir}xenobot_{xenobot_id}_features_CEC.csv', index = False)


if __name__ == '__main__':
    main()