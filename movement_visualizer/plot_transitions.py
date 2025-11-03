import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import circvar
from collections import Counter
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

try: 
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri, r, StrVector, IntVector, FloatVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter, converter 

    # converts python df to R df
    def py_to_r_df(py_df):
        with localconverter(converter + pandas2ri.converter):
            r_df = pandas2ri.py2rpy(py_df)
        return r_df

    # converts R df to python df
    def r_to_py_df(r_df):
        with localconverter(converter + pandas2ri.converter):
            py_df = pandas2ri.rpy2py(r_df)
        return py_df
        
    # converts python array to R matrix
    def py_to_r_matrix(np_array):
        with localconverter(converter, numpy2ri.converter):
            r_matrix = ro.r.matrix(
                np_array, 
                nrow = np_array.shape[0], 
                ncol = np_array.shape[1]
            )
        return r_matrix
        
except ImportError as e:
    raise ImportError("Error: rpy2 is not installed") from e


# makes sure that cec package is installed
def set_up_r_packages():
    try:
        utils = importr('utils')
        installed_packages = ro.r('installed.packages()')
        package_names = list(installed_packages[0])

        if not 'CEC' in package_names:
            utils.chooseCRANmirror(ind = 1)
            utils.install_packages('CEC')
            
        r_cec = importr('CEC')
        return r_cec
    except Exception as e:
        raise ImportError("Error: could not set up R package CEC") from e

r_cec = set_up_r_packages()


# break the video up into chunks (30-sec based on research paper)
# TODO: adjust to 30 later, using 15 for testing for now
def break_into_chunks(df, chunk_duration = 15 * 30):
    chunks = []
    file_id = df.attrs.get('file_id', 'unknown')

    # group based on xenobot id (fixed track column)
    for xenobot_id, group in df.groupby('track_fixed'):
        group = group.sort_values('frame')
        start = group['frame'].min()
        end = group['frame'].max()

        # per start of chunk 
        for chunk_start in range(start, end, chunk_duration):
            chunk_end = chunk_start + chunk_duration
            chunk_data = group[(group['frame'] >= chunk_start) & (group['frame'] < chunk_end)]

            # add chunks to list
            if len(chunk_data) > 10:
                chunks.append({
                    'xenobot_id': xenobot_id,
                    'file_id': file_id,
                    'chunk_start': chunk_start,
                    'chunk_end': chunk_end,
                    'data': chunk_data
                })

    return chunks

# calculates & adds calculations for linear speed, heading, angular speed to csv file
def calculate_metrics(chunk):
    data = chunk['data']

    if len(data) < 2:
        return None
    
    features = {
        'xenobot_id': chunk['xenobot_id'],
        'file_id': chunk['file_id'],
        'chunk_start': chunk['chunk_start'],
        'mean_speed': data['linear_speed'].mean(),
        'mean_angular_speed': data['angular_speed'].mean(),
        'straightness': 1 - circvar(data['heading'].dropna())
    }

    # TODO: ask abt the gyration index (does not match paper but ts makes sense)
    if not data['angular_speed'].isna().all():
        features['gyration'] = 1 - circvar(data['angular_speed'].dropna())
    else:
        features['gyration'] = 0

    return features

# assigns each xenobot a unique id across the different files
def create_global_ids(features_df):
    features_df['new_id_combo'] = features_df['file_id'] + '-' + features_df['xenobot_id'].astype(str)
    unique_ids = features_df['new_id_combo'].unique()
    new_unique_ids = {id : index + 1 for index, id in enumerate(sorted(unique_ids))}
    features_df['new_id'] = features_df['new_id_combo'].map(new_unique_ids)

    features_df.drop('new_id_combo', axis = 1, inplace = True)
    return features_df

# converts numpy to r matrix
def numpy_to_r_matrix(np_array):
    np_array = np_array.astype(float)
    nr, nc = np_array.shape
    r_matrix = r.matrix(
        FloatVector(np_array.T.flatten()), 
        nrow = nr,
        ncol = nc
    )

    return r_matrix

# TODO: make sure inputs are correct

def fit_cec_clustering(features_df, n_states = 4, cov_type = 'spherical', iterations = 25, random_seed = 42):
    features = ['mean_speed', 'straightness', 'gyration', 'mean_angular_speed']

    x = features_df[features].fillna(0)
    np.random.seed(random_seed)
    r('set.seed({})'.format(random_seed))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    try:
        r_data = numpy_to_r_matrix(x_scaled)
    except Exception as e:
        return None
    
    cov_param = StrVector([cov_type])
    card_min_value = max(1, int(0.05 * len(x_scaled))) 
    card_min = IntVector([card_min_value])

    try:
        result = r['cec'](
            x = r_data,
            centers = n_states,
            type = cov_param,
            iter_max = iterations,
            card_min = card_min
        )

        try:
            result_names = list(result.names) if hasattr(result, 'names') else []
        except Exception as e:
            result_names = []

        cluster_key = next((k for k in ('clusters', 'cluster', 'cluster.indices', 'cluster_idx') if k in result_names), None)
        centers_key = next((k for k in ('centers', 'centroids') if k in result_names), None)
        cov_key = next((k for k in ('cov', 'covariances', 'covariances.model', 'covs') if k in result_names), None)
        cost_key = next((k for k in ('cost', 'cost.function', 'cost.function.value') if k in result_names), None)

        missing = []
        if cluster_key is None:
            missing.append('clusters/cluster')
        if centers_key is None:
            missing.append('centers')
        if cov_key is None:
            missing.append('cov/covariances')
        if cost_key is None:
            missing.append('cost')

        if missing:
            return None

        clusters_raw = np.array(result.rx2(cluster_key))
        centers_raw = np.array(result.rx2(centers_key))
        cov_matrices = result.rx2(cov_key)
        energy = float(result.rx2(cost_key)[0])

        unique_labels = np.unique(clusters_raw)

        # align centers shape, correct the orientation of matrix
        centers = centers_raw
        try:
            if centers.shape[0] != len(unique_labels) and centers.shape[1] == len(unique_labels):
                centers = centers.T
        except Exception:
            pass

        if centers.shape[0] != len(unique_labels):
            computed_centers = []

            for lbl in sorted(unique_labels):
                mask = (clusters_raw == lbl)

                if mask.sum() > 0:
                    computed_centers.append(x_scaled[mask].mean(axis=0))
                else:
                    computed_centers.append(np.zeros(x_scaled.shape[1]))
            centers = np.vstack(computed_centers)

        label_map = {old: i for i, old in enumerate(sorted(unique_labels))}
        clusters = np.array([label_map[l] for l in clusters_raw])

        centers_ordered = centers.copy()
        return clusters, centers_ordered, cov_matrices, scaler
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

# TODO: get threshold views
def label_states(features_df):
    features = ['mean_speed', 'straightness', 'gyration', 'mean_angular_speed']
   
    state_means = features_df.groupby('state')[features].mean()

    # percentiles for adaptive thresholding
    speed_25 = features_df['mean_speed'].quantile(0.25)
    speed_75 = features_df['mean_speed'].quantile(0.75)
    straight_25 = features_df['straightness'].quantile(0.25)
    straight_75 = features_df['straightness'].quantile(0.75)

    gyration_range = features_df['gyration'].max() - features_df['gyration'].min()
    use_gyration = gyration_range > 0.01

    labels = {}
    for state in state_means.index:
        mean_speed = state_means.loc[state, 'mean_speed']
        straightness = state_means.loc[state, 'straightness']
        mean_angular_speed = state_means.loc[state, 'mean_angular_speed']
  
        if mean_speed < speed_25:
            labels[state] = 'idle'
        elif straightness > straight_75:
            labels[state] = 'linear'
        elif straightness < straight_25:
            labels[state] = 'circling'
        else:
            labels[state] = 'intermediate'

    return labels

def apply_cec_clustering(features_df, clusters, scaler):
    features_df['state'] = clusters
    state_labels = label_states(features_df) 
    features_df['state_label'] = features_df['state'].map(state_labels)

    return features_df

def calculate_transition_probabilities(features_df):
    transitions = []

    for id in features_df['new_id'].unique():
        bot_data = features_df[features_df['new_id'] == id].sort_values('chunk_start')

        for i in range(len(bot_data) - 1):
            curr_state = bot_data.iloc[i]['state_label']
            next_state = bot_data.iloc[i + 1]['state_label']
            transitions.append((curr_state, next_state))

    transition_counts = Counter(transitions)
    states = sorted(features_df['state_label'].unique())
    transition_matrix = pd.DataFrame(0, index = states, columns = states, dtype = float)

    for from_state in states:
        total_from = sum(count for (s1, s2), count in transition_counts.items() if s1 == from_state)

        if total_from > 0:
            for to_state in states:
                count = transition_counts.get((from_state, to_state), 0)
                transition_matrix.loc[from_state, to_state] = count / total_from

    return transition_matrix


def plot_transition_heat_map(transition_matrix, title = "Xenobot Movement State Transition Probabilities", save_path = None):
    plt.figure(figsize = (8, 6))
    sns.heatmap(transition_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', vmin = 0, vmax = 1)
    plt.title(title)
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 300)

    plt.show()

# plot as graphs w/ vertices + edges
def plot_transition_graph(transition_matrix, title = "Xenobot Movement State Transition Probabilities"):
    G = nx.DiGraph()

    for state in transition_matrix.index:
        G.add_node(state)

    for from_state in transition_matrix.index:
        for to_state in transition_matrix.columns:
            prob = transition_matrix.loc[from_state, to_state]
            if prob > 0.001:
                G.add_edge(from_state, to_state, weight=prob)

    pos = nx.circular_layout(G)

    plt.figure(figsize = (10, 10))
    nx.draw_networkx_nodes(G, pos, node_color = 'lightblue', node_size = 3000, edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size = 12, font_weight='bold')

    regular_edges = [(u, v, d) for (u, v, d) in G.edges(data=True) if u != v]
    self_loops = [(u, v, d) for (u, v, d) in G.edges(data=True) if u == v]

    for (u, v, d) in regular_edges:
        weight = d['weight']
        width = 5 * weight
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[(u,v)], 
            width = width,
            arrows = True, 
            arrowstyle = '-|>', 
            arrowsize = 15,
            connectionstyle = "arc3, rad = 0.2", 
            alpha = 0.8,
            min_source_margin = 50,
            min_target_margin = 50
        )

    for (u, v, d) in self_loops:
        weight = d['weight']
        width = 5 * weight
        x, y = pos[u]
        
        circle = plt.Circle(
            (x, y + 0.15),
            0.08,
            color = 'black',
            fill = False,
            linewidth = width,
            alpha = 0.8
        )
        plt.gca().add_patch(circle)
        
        arrow = FancyArrowPatch(
            (x + 0.08, y + 0.15),
            (x + 0.06, y + 0.12),
            arrowstyle = '-|>',
            mutation_scale = 15,
            color = 'black',
            linewidth = width,
            alpha = 0.8 
        )
        plt.gca().add_patch(arrow)

    for (u, v, d) in regular_edges:
        x_mid = (pos[u][0] + pos[v][0]) / 2
        y_mid = (pos[u][1] + pos[v][1]) / 2
        plt.text(
            x_mid,
            y_mid,
            f"{d['weight']*100:.1f}%",
            ha = 'center',
            fontsize = 9,
            color = 'red',
            bbox = dict(
                boxstyle = 'round, pad = 0.3',
                facecolor = 'white',
                alpha = 0.7
            )
        )
    
    for (u, v, d) in self_loops:
        x, y = pos[u]
        plt.text(
            x, 
            y + 0.25, 
            f"{d['weight']*100:.1f}%",
            ha='center', 
            fontsize=10, 
            color='black',
            bbox = dict(
                boxstyle = 'round, pad = 0.3',
                facecolor = 'white',
                alpha = 0.7
            )
        )

    plt.axis('off')
    plt.title(title, fontsize = 14)
    plt.tight_layout()
    plt.show()


def analyze_combined_data(csv_files, n_states = 4, cov_type = 'diagonal', random_seed=42):
    all_features = []

    for i, path in enumerate(csv_files):
        df = pd.read_csv(path)
        df.attrs['file_id'] = f"file_{i + 1}"

        blocks = break_into_chunks(df)
        
        features = [calculate_metrics(block) for block in blocks]
        features_clean = [f for f in features if f is not None]
        
        all_features.extend(features_clean)

    all_features_df = pd.DataFrame(all_features)
    all_features_df = create_global_ids(all_features_df)

    features = ['mean_speed', 'straightness', 'gyration', 'mean_angular_speed']
    
    result = fit_cec_clustering(
        all_features_df,
        n_states = n_states,
        cov_type = cov_type,
        random_seed = random_seed
    )
    
    if result is None:
        print("Error: CEC clustering failed")
        return None
    
    clusters, centers, cov_matrices, scaler = result
    all_features_df = apply_cec_clustering(all_features_df, clusters, scaler)
    state_counts = all_features_df['state_label'].value_counts()
  

    for file_id in sorted(all_features_df['file_id'].unique()):
        file_counts = all_features_df[all_features_df['file_id'] == file_id]['state_label'].value_counts()
    
    transition_matrices = []
    weights = []

    for file_id in sorted(all_features_df['file_id'].unique()):
        file_features_df = all_features_df[all_features_df['file_id'] == file_id].copy()

        if len(file_features_df) > 0:
            tm = calculate_transition_probabilities(file_features_df)
            transition_matrices.append(tm)
            weights.append(len(file_features_df))

    all_states = sorted(all_features_df['state_label'].unique())
    normalized_matrices = []

    for tm in transition_matrices:
        tm_normalized = tm.reindex(index = all_states, columns = all_states, fill_value = 0)
        normalized_matrices.append(tm_normalized)

    combined_matrix = sum(tm * w for tm, w in zip(normalized_matrices, weights)) / sum(weights)

    plot_transition_heat_map(
        combined_matrix,
        title = f'Combined Transition Probabilities (CEC, n = {n_states})'
    )

    plot_transition_graph(
        combined_matrix,
        title = f'Combined State Transitions (CEC, n = {n_states})'
    )

    return all_features_df, combined_matrix, clusters, centers, cov_matrices, scaler


if __name__ == '__main__':
    csv_files = [
        "/Users/yuxin/Downloads/bot_swarms/processed/2025-01-22-5_fixed_processed.csv",
        "/Users/yuxin/Downloads/bot_swarms/processed/2025-01-22-4_fixed_processed.csv",
        "/Users/yuxin/Downloads/bot_swarms/processed/2025-01-22-3_fixed_processed.csv",
        "/Users/yuxin/Downloads/bot_swarms/processed/2025-01-22-2_fixed_processed.csv",
        "/Users/yuxin/Downloads/bot_swarms/processed/2025-01-22-1_fixed_processed.csv",
    ]

    # Set random_seed for reproducible results
    # Change this number to explore different clustering solutions
    features_df, transition_matrix, clusters, centers, cov_matrices, scaler = analyze_combined_data(
        csv_files,
        n_states = 4,
        cov_type = 'diagonal',
        random_seed = 42  # Use same seed for reproducible results
    )

    if features_df is not None:
        features_df.to_csv('all_features_with_cec_states.csv', index = False)
        transition_matrix.to_csv('combined_transition_matrix_cec.csv')
        
        # Save cluster centers in original scale
        features = ['mean_speed', 'straightness', 'gyration', 'mean_angular_speed']
        centers_original = scaler.inverse_transform(centers)
        centers_df = pd.DataFrame(centers_original, columns = features)
        centers_df.to_csv('cec_cluster_centers.csv', index = False)
