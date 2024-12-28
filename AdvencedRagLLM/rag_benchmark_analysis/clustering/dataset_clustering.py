from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Dict
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import  file_operations_utils as fo_utils

class DataClustering:
    def __init__(self):
        self.embedding: np.ndarray = None
        self.data: List[str] = None
        self.df: pd.DataFrame = None

            
    def generate_embedding(self, data: List[str], embedding_model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr") -> np.ndarray:

        if not isinstance(data, list):
            raise ValueError("Input data must be a list of strings or a list of lists containing strings.")

        # Check if it is a list of lists containing strings
        if all(isinstance(item, list) and all(isinstance(sub_item, str) for sub_item in item) for item in data):
            # Flatten the list of lists
            data = [sub_item for item in data for sub_item in item]
        elif not all(isinstance(item, str) for item in data):
            # If not all elements are strings or valid lists, raise an error
            raise ValueError("Input must be a one-dimensional list of strings or a list of lists containing strings.")

        # Ensure the list is not empty after validation
        if not data:
            raise ValueError("Input data contains no valid strings to process.")

        self.data = data
        self.df = pd.DataFrame(self.data, columns=["TEXT"])
        embedding_model = SentenceTransformer(embedding_model_name)
        embedding = embedding_model.encode(data, show_progress_bar=True)
        self.embedding = embedding
        return embedding
        

    def get_best_metrics(self) -> Dict[str, float]:
        """Finds the best eps and min_samples values using silhouette score."""
        print("Finding best metrics...")
        eps_values = np.arange(0.05, 4.05, 0.05)  # Finer granularity
        min_samples_values = range(2, 6)         # Wider range

        results = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                labels = dbscan.fit_predict(self.embedding)

                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_count = np.sum(labels == -1)

                if num_clusters > 1:
                    try:
                        sil_score = silhouette_score(self.embedding, labels, metric='cosine')
                    except:
                        sil_score = None
                else:
                    sil_score = None

                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'num_clusters': num_clusters,
                    'noise_count': noise_count,
                    'silhouette_score': sil_score
                })

        sorted_results = sorted(
        (r for r in results if r['silhouette_score'] is not None),
        key=lambda x: x['silhouette_score'],
        reverse=True
        )
        return sorted_results


    def perform_clustering(self, eps: Optional[float] = None, min_samples: Optional[int] = None, metrics:dict=None):
        """Performs DBSCAN clustering with optional custom parameters."""
        if eps is None or min_samples is None:
            eps = metrics["eps"]
            min_samples = metrics["min_samples"]
            print(f"Using best parameters: ")
        else:
            print(f"Using provided parameters: ")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(self.embedding)
        self.df['cluster'] = cluster_labels
        sil_score = silhouette_score(self.embedding, cluster_labels, metric='cosine')
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_count = np.sum(cluster_labels == -1)
        parameters_results = {
            'eps': eps,
            'min_samples': min_samples,
            'num_clusters': num_clusters,
            'noise_count': noise_count,
            'silhouette_score': sil_score
        }
        
        return self.df, parameters_results

    def save_cluster_summary(self, output_file_path, metrics) -> None:
        # Count occurrences within clusters
        """
        Generates and saves a summary of clusters for the selected embedding type.

        The summary contains two columns: 'count' and 'unique_values'.
        'count' is the number of occurrences of each cluster.
        'unique_values' is a list of unique values within each cluster.

        Parameters
        ----------
        output_file_path : str
            The path to the output file.

        Returns
        -------
        None
        """
        cluster_counts = self.df.groupby('cluster')["TEXT"].count().reset_index()
        cluster_counts = cluster_counts.rename(columns={"TEXT": 'count'})

        # List unique values within clusters
        text_values = self.df.groupby('cluster')["TEXT"].apply(lambda x: list((x))).reset_index()

        # Merge and save the summary
        summary = pd.merge(cluster_counts, text_values, on='cluster', how='left')
        summary_dict_list = summary.to_dict(orient='records')
        summary_dict_list.append(summary_dict_list.pop(0))
        summary_dict_list.insert(0, metrics)
        fo_utils.write_textual_file(summary_dict_list, output_file_path)
        print("saved: ",output_file_path )

    def plot_k_distance(self, output_file_path, k: int = 2):
        """Plots the k-distance graph to find the best epsilon value."""
        neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors_fit = neighbors.fit(self.embedding)
        distances, _ = neighbors_fit.kneighbors(self.embedding)
        distances = np.sort(distances[:, k - 1], axis=0)
        plt.plot(distances)
        plt.ylabel("k-distance")
        plt.xlabel("Sorted Points")
        plt.title(f"k-Distance Graph for k={k}")
        plt.savefig(output_file_path)


    def save_cluster_visualization(self, output_file_path):
        """Visualizes clusters in 2D using t-SNE."""
        embeddings_array = np.array(self.embedding)

        # Reduce embeddings to 2D
        tsne = TSNE(n_components=2, metric='cosine', random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_array)

        # Visualize with cluster labels
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            hue=self.df['cluster'],
            palette='Set1',
            legend="full"
        )
        plt.title("t-SNE Visualization of Clusters")
        plt.savefig(output_file_path)

def generate_data_clustring(task_name:str,base_version:str,data:list,start_best_metric_order:int=0,stop_best_metric_order:int=40,range_order:int=3 ):
    current_dir = os.path.dirname(os.path.abspath(__file__))
   
    
    clustering = DataClustering()
    
    clustering.generate_embedding(data)
    #clustering.plot_k_distance(output_dbscan_eps_k_distance_graph,k=5)
    sorted_metrics = clustering.get_best_metrics()
    response_list = []
    for metric_order in range(start_best_metric_order,stop_best_metric_order,range_order):
        version = f"{base_version}_{metric_order}"
        output_file_folder = os.path.join(current_dir,"dataset_clustering_results",task_name,version)
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
            print("Created  folder: ", output_file_folder, flush=True)
        else:
            print("Output folder already exists for this task and version")
        output_csv_file_path = os.path.join(output_file_folder, f"clustered_data_{task_name}_{version}.csv")
        output_json_file_path = os.path.join(output_file_folder, f"clustered_data_{task_name}_{version}.json")
        #output_dbscan_eps_k_distance_graph = os.path.join(output_file_folder, f"clustered_data_{task_name}_{version}_eps-k-distance.png")
        output_cluster_result_graph = os.path.join(output_file_folder, f"clustered_data_{task_name}_{version}_t-SNE.png")
        
        metrics = sorted_metrics[metric_order]
        clustering_df, parameter_results= clustering.perform_clustering(metrics=metrics)
        print(parameter_results)
        #save step
        fo_utils.write_textual_file(clustering_df, output_csv_file_path)
        clustering.save_cluster_summary(output_json_file_path, metrics)
        clustering.save_cluster_visualization(output_cluster_result_graph)
        response_list.append(f"Data clustered results saved at {output_file_folder}")
    return response_list

    
