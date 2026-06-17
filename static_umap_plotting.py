from collections.abc import Mapping
from pathlib import Path
import re

def extract_umap_info(x,y,base_directory="/work/hdd/bemi/dmiura/data_downloads/tng100_snap72/hyrax_runs"):
  """Extract UMAP directory from 3dumap output file."""

  # Construct paths
  run_dir = Path(base_directory) / f"run{x}"
  dumap_name = f"udb{x}_{y}"

  dumap_output_file = run_dir / f"{dumap_name}.txt"
  dumap_toml_file = run_dir / f"{dumap_name}.toml"
    
  try:
      with open(dumap_output_file, 'r') as f:
          content = f.read()

      # Look for the UMAP results save pattern
      pattern = r'Saving UMAP results to (.+)'
      match = re.search(pattern, content)

      if match:
          umap_dir = match.group(1).strip()
          return umap_dir, dumap_toml_file
      else:
          raise ValueError(f"Could not find UMAP directory in {dumap_output_file}")

  except FileNotFoundError:
      raise FileNotFoundError(f"3dumap output file not found: {dumap_output_file}")
  

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

def plot_umap_simple(ax, config=None, input_dir=None, alpha=0.6, s=1, color=None,
                       color_column=None, vmin=None, vmax=None, log_colorbar=False,
                       density=False, cmap='viridis', title=None, supress_hyrax_logs=False):
  """
  Create a simple static matplotlib scatter plot of 2D UMAP results on provided axis.
  
  Parameters
  ----------
  ax : matplotlib.axes.Axes
      The axis to plot on
  config: 
      Hyrax config object
  input_dir : str or Path, optional
      Directory containing UMAP results. If None, uses most recent in current results dir.
  alpha : float, default 0.6
      Point transparency
  s : float, default 1
      Point size
  color : array-like, optional
      Color values for scatter plot. If provided, overrides color_column.
  color_column : str, optional
      Name of catalog column to use for coloring points
  vmin : float, optional
      Minimum value for color scaling
  vmax : float, optional
      Maximum value for color scaling
  log_colorbar : bool, default False
      If True, use logarithmic color scaling
  density : bool, default False
      If True, plot hexbin density plot instead of scatter
  cmap : str, default 'viridis'
      Colormap name
  title : str, default None
      Plot title
      
  Returns
  -------
  matplotlib.axes.Axes
      The axis object with the plot
  """

  from hyrax.data_sets.inference_dataset import InferenceDataSet
  from matplotlib.colors import LogNorm, Normalize
      
  # Load UMAP results
  if supress_hyrax_logs is True:
      logging.disable(logging.CRITICAL)
  umap_results = InferenceDataSet(config, results_dir=input_dir, verb="umap")
  logging.disable(logging.NOTSET)
    
  # Extract 2D coordinates
  points = np.array([point.numpy() for point in umap_results])
  x, y = points[:, 0], points[:, 1]

  # Handle color specification
  color_values = None
  colorbar_label = None

  if color is not None:
      # Use provided color array (original behavior)
      color_values = color
      colorbar_label = 'Point Index'
  elif color_column is not None:
      # Get color values from catalog column
      try:
          # Get all available fields to check if column exists
          available_fields = umap_results.metadata_fields()
          if color_column not in available_fields:
              raise ValueError(f"Column '{color_column}' not found in dataset. Available fields: {available_fields}")

          # Get all indices for the dataset
          all_indices = list(range(len(umap_results)))

          # Extract metadata for the specified column
          metadata = umap_results.metadata(all_indices, [color_column])
          color_values = np.array(metadata[color_column])
          colorbar_label = color_column

      except Exception as e:
          print(f"Warning: Could not load column '{color_column}': {e}")
          print("Proceeding without coloring")

  if density:
    # Create hexbin plot
    if color_values is not None:
        # Hexbin with color values - compute mean/median of color values in each bin
        # Determine normalization for color values
        if log_colorbar:
            # Ensure positive values for log scaling
            if np.any(color_values <= 0):
                min_positive = np.min(color_values[color_values > 0]) if np.any(color_values > 0) else 1e-10
                color_values = np.maximum(color_values, min_positive)
                print(f"Warning: Non-positive values found, clamped to {min_positive} for log scaling")

            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Use reduce_C_function to aggregate color values in each hexbin
        hb = ax.hexbin(x, y, C=color_values, gridsize=50, cmap=cmap, norm=norm, reduce_C_function=np.mean)
        cbar = plt.colorbar(hb, ax=ax, label=colorbar_label)

        if log_colorbar:
            cbar.set_label(f'{colorbar_label} (log scale)')
    else:
        # Hexbin with just point density (counts)
        norm = LogNorm() if log_colorbar else None
        hb = ax.hexbin(x, y, gridsize=50, cmap=cmap, norm=norm)
        plt.colorbar(hb, ax=ax, label='Count')
  else:
    # Regular scatter plot
    if color_values is not None:
        # Determine normalization
        if log_colorbar:
            # Ensure positive values for log scaling
            if np.any(color_values <= 0):
                min_positive = np.min(color_values[color_values > 0]) if np.any(color_values > 0) else 1e-10
                color_values = np.maximum(color_values, min_positive)
                print(f"Warning: Non-positive values found, clamped to {min_positive} for log scaling")

            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        scatter = ax.scatter(x, y, alpha=alpha, s=s, c=color_values, cmap=cmap, norm=norm)
        cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label)
    else:
        scatter = ax.scatter(x, y, alpha=alpha, s=s)


  # Styling
  if title is not None:
      ax.set_title(title)

  return ax


def plot_multiple_umaps(runs, expts, figsize=None, ncols=3, density=False, 
                          cmap='viridis', alpha=0.6, s=1, save_path=None, 
                          suptitle="UMAP Comparisons", dpi=150, supress_hyrax_logs=False,
                          color=None, color_column=None, vmin=None, vmax=None, log_colorbar=False):
      """
      Create a multi-panel plot of UMAP results from multiple run/experiment combinations.
      
      Parameters
      ----------
      runs : array-like
          Array of run numbers
      expts : array-like
          Array of experiment numbers (same length as runs)
      figsize : tuple, optional
          Figure size. If None, calculated based on number of plots
      ncols : int, default 3
          Number of columns in the subplot grid
      density : bool, default False
          If True, plot hexbin density plots instead of scatter
      cmap : str, default 'viridis'
          Colormap name
      alpha : float, default 0.6
          Point transparency (for scatter plots)
      s : float, default 1
          Point size (for scatter plots)
      save_path : str or Path, optional
          Path to save the plot. If None, just displays it
      suptitle : str, default "UMAP Comparisons"
          Overall figure title
      dpi : int, default 150
          DPI for the figure
      color_column : str, optional
          Name of catalog column to use for coloring points
      vmin : float, optional
          Minimum value for color scaling
      vmax : float, optional
          Maximum value for color scaling
      log_colorbar : bool, default False
          If True, use logarithmic color scaling
          
      Returns
      -------
      matplotlib.figure.Figure
          The matplotlib figure object
      """
      import math
      import logging
      from tqdm.notebook import tqdm

      # Ensure runs and expts are the same length
      if len(runs) != len(expts):
          raise ValueError("runs and expts arrays must be the same length")

      n_plots = len(runs)
      nrows = math.ceil(n_plots / ncols)

      # Calculate figure size if not provided
      if figsize is None:
          figsize = (ncols * 4, nrows * 3)

      # Create figure and subplots
      fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)

      # Handle case where we have only one row or column
      if n_plots == 1:
          axes = [axes]
      elif nrows == 1:
          axes = axes.flatten()
      elif ncols == 1:
          axes = axes.flatten()
      else:
          axes = axes.flatten()

      # Plot each run/experiment combination
      for i, (run, expt) in enumerate(tqdm(zip(runs, expts),total=len(expts))):
          if i >= len(axes):
              break

          ax = axes[i]

          try:
              # Get UMAP info and create config
              umap_dir, config_file = extract_umap_info(run, expt)
              if supress_hyrax_logs is True:
                  logging.disable(logging.CRITICAL)
              import hyrax
              h = hyrax.Hyrax(config_file=config_file)
              logging.disable(logging.NOTSET)
              
              # Create title
              title = f"Run {run}, Expt {expt}"

              # Plot on this axis
              plot_umap_simple(ax, config=h.config, input_dir=umap_dir,
                             alpha=alpha, s=s, density=density, cmap=cmap, title=title,
                             supress_hyrax_logs=supress_hyrax_logs,
                             color=color, color_column=color_column, vmin=vmin, vmax=vmax,
                             log_colorbar=log_colorbar)

          except Exception as e:
              # If there's an error, show it on the plot
              ax.text(0.5, 0.5, f"Error loading\nRun {run}, Expt {expt}\n{str(e)}",
                     ha='center', va='center', transform=ax.transAxes)
              ax.set_title(f"Run {run}, Expt {expt} - ERROR")

      # Hide unused subplots
      for i in range(n_plots, len(axes)):
          axes[i].axis('off')

      # Set overall title
      fig.suptitle(suptitle, fontsize=16)

      # Adjust layout
      plt.tight_layout()

      # Save or show
      if save_path:
          plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
      else:
          plt.show()

      return fig


def _load_toml(path):
    """Load a TOML file without requiring one specific TOML package."""
    path = Path(path)
    try:
        import tomllib

        with path.open("rb") as handle:
            return tomllib.load(handle)
    except ModuleNotFoundError:
        try:
            import tomli

            with path.open("rb") as handle:
                return tomli.load(handle)
        except ModuleNotFoundError:
            import toml

            with path.open("r") as handle:
                return toml.load(handle)


def extract_umap_and_inference_info(
    x,
    y,
    base_directory="/work/hdd/bemi/dmiura/data_downloads/tng100_snap72/hyrax_runs",
):
    """Extract UMAP, inference, and config paths for a run/experiment pair."""
    umap_dir, config_file = extract_umap_info(x, y, base_directory=base_directory)
    config = _load_toml(config_file)

    try:
        inference_dir = config["results"]["inference_dir"]
    except KeyError as exc:
        raise ValueError(
            f"No [results].inference_dir entry found in {config_file}"
        ) from exc

    return umap_dir, inference_dir, config_file


def _result_batch_files(results_dir):
    results_dir = Path(results_dir)
    batch_files = sorted(
        file
        for file in results_dir.glob("batch_*.npy")
        if re.fullmatch(r"batch_\d+\.npy", file.name)
    )
    if not batch_files:
        raise FileNotFoundError(
            f"No batch_<number>.npy files found in {results_dir}"
        )
    return batch_files


def _batch_field(batch, field_name, batch_file):
    if hasattr(batch, "files"):
        available = list(batch.files)
        if field_name not in available:
            raise KeyError(
                f"Field '{field_name}' not found in {batch_file}. "
                f"Available fields: {available}"
            )
        return batch[field_name]

    available = batch.dtype.names or ()
    if field_name not in available:
        raise KeyError(
            f"Field '{field_name}' not found in {batch_file}. "
            f"Available fields: {list(available)}"
        )
    return batch[field_name]


def load_result_tensors(
    results_dir,
    flatten=True,
    tensor_key="tensor",
    id_key="id",
):
    """
    Load Hyrax result tensors directly from batch files.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Directory containing Hyrax `batch_<number>.npy` files.
    flatten : bool, default True
        Flatten each tensor to one row per object. This is usually what
        clustering and distance metrics expect.
    tensor_key : str, default "tensor"
        Structured-array field containing embeddings.
    id_key : str, default "id"
        Structured-array field containing object IDs. If absent, IDs are
        returned as None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Embeddings and object IDs in batch order.
    """
    tensors = []
    object_ids = []
    found_ids = True

    for batch_file in _result_batch_files(results_dir):
        batch = np.load(batch_file, allow_pickle=False)
        tensors.append(_batch_field(batch, tensor_key, batch_file))

        try:
            object_ids.append(_batch_field(batch, id_key, batch_file))
        except KeyError:
            found_ids = False

    data = np.concatenate(tensors, axis=0)
    if flatten:
        data = data.reshape(data.shape[0], -1)

    ids = np.concatenate(object_ids, axis=0) if found_ids else None
    return data, ids


def _normalize_result_ids(ids):
    if ids is None:
        return None
    return np.asarray(ids).reshape(-1).astype(str)


def align_tensors_by_id(
    primary_tensors,
    primary_ids,
    secondary_tensors,
    secondary_ids,
    missing="raise",
):
    """
    Align two embedding arrays by object ID.

    The primary array controls output order. Set `missing="drop"` to keep only
    IDs present in both arrays.
    """
    if missing not in {"raise", "drop"}:
        raise ValueError("missing must be either 'raise' or 'drop'")

    primary_ids = _normalize_result_ids(primary_ids)
    secondary_ids = _normalize_result_ids(secondary_ids)

    if primary_ids is None or secondary_ids is None:
        if len(primary_tensors) != len(secondary_tensors):
            raise ValueError(
                "Cannot align results without IDs when array lengths differ: "
                f"{len(primary_tensors)} vs {len(secondary_tensors)}"
            )
        return primary_tensors, secondary_tensors, primary_ids

    if len(np.unique(primary_ids)) != len(primary_ids):
        raise ValueError("Primary result IDs contain duplicates; cannot align safely")
    if len(np.unique(secondary_ids)) != len(secondary_ids):
        raise ValueError("Secondary result IDs contain duplicates; cannot align safely")

    secondary_lookup = {
        object_id: index for index, object_id in enumerate(secondary_ids)
    }
    primary_indexes = []
    secondary_indexes = []
    missing_ids = []

    for primary_index, object_id in enumerate(primary_ids):
        secondary_index = secondary_lookup.get(object_id)
        if secondary_index is None:
            missing_ids.append(object_id)
            continue
        primary_indexes.append(primary_index)
        secondary_indexes.append(secondary_index)

    if missing_ids and missing == "raise":
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"{len(missing_ids)} primary IDs are missing from the secondary "
            f"results. First missing IDs: {preview}"
        )
    if not primary_indexes:
        raise ValueError("No overlapping object IDs found between result arrays")

    primary_indexes = np.asarray(primary_indexes)
    secondary_indexes = np.asarray(secondary_indexes)
    return (
        primary_tensors[primary_indexes],
        secondary_tensors[secondary_indexes],
        primary_ids[primary_indexes],
    )


def _as_2d_numeric_array(data, name="data"):
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if data.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return data


def _prepare_metric_data(data, standardize=False):
    data = _as_2d_numeric_array(data)
    scaler = None

    if standardize:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data, scaler


def _median_centers_from_labels(data, labels):
    labels = np.asarray(labels)
    valid_labels = np.array([label for label in np.unique(labels) if label != -1])

    if len(valid_labels) == 0:
        return None, valid_labels

    centers = []
    for label in valid_labels:
        centers.append(np.median(data[labels == label], axis=0))

    return np.asarray(centers), valid_labels


def _cluster_quality_metrics(data, labels, metric="euclidean", clusterer=None):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    valid_cluster_labels = np.array(
        [label for label in unique_labels if label != -1]
    )
    non_noise_mask = labels != -1
    n_noise = int(np.sum(~non_noise_mask))

    metrics = {
        "n_samples": int(len(labels)),
        "n_clusters": int(len(valid_cluster_labels)),
        "n_noise_points": n_noise,
        "noise_fraction": float(n_noise / len(labels)),
        "label_counts": {
            str(label): int(np.sum(labels == label)) for label in unique_labels
        },
    }

    if len(valid_cluster_labels) > 1 and np.sum(non_noise_mask) > len(valid_cluster_labels):
        from sklearn.metrics import silhouette_score

        metrics["silhouette_score"] = float(
            silhouette_score(
                data[non_noise_mask],
                labels[non_noise_mask],
                metric=metric,
            )
        )

        if metric == "euclidean":
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

            metrics["calinski_harabasz_score"] = float(
                calinski_harabasz_score(data[non_noise_mask], labels[non_noise_mask])
            )
            metrics["davies_bouldin_score"] = float(
                davies_bouldin_score(data[non_noise_mask], labels[non_noise_mask])
            )

    if clusterer is not None and hasattr(clusterer, "inertia_"):
        metrics["inertia"] = float(clusterer.inertia_)

    if clusterer is not None and hasattr(clusterer, "aic"):
        metrics["aic"] = float(clusterer.aic(data))
    if clusterer is not None and hasattr(clusterer, "bic"):
        metrics["bic"] = float(clusterer.bic(data))

    return metrics


def _fit_clustering(data, method, n_clusters, metric, params):
    method = method.lower()
    params = dict(params)

    if method == "kmeans":
        from sklearn.cluster import KMeans

        clusterer_params = {
            "n_clusters": n_clusters,
            "random_state": 42,
            "n_init": 10,
        }
        clusterer_params.update(params)
        clusterer = KMeans(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers = clusterer.cluster_centers_
        center_labels = np.arange(len(centers))

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN

        clusterer_params = {
            "eps": 0.5,
            "min_samples": 5,
            "metric": metric,
        }
        clusterer_params.update(params)
        clusterer = DBSCAN(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers, center_labels = _median_centers_from_labels(data, labels)

    elif method == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN requires a scikit-learn version that provides "
                "sklearn.cluster.HDBSCAN"
            ) from exc

        clusterer_params = {
            "min_cluster_size": 5,
            "min_samples": None,
            "metric": metric,
        }
        clusterer_params.update(params)
        clusterer = HDBSCAN(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers, center_labels = _median_centers_from_labels(data, labels)

    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        clusterer_params = {
            "n_components": n_clusters,
            "random_state": 42,
        }
        clusterer_params.update(params)
        clusterer = GaussianMixture(**clusterer_params)
        clusterer.fit(data)
        labels = clusterer.predict(data)
        centers = clusterer.means_
        center_labels = np.arange(len(centers))

    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        clusterer_params = {
            "n_clusters": n_clusters,
            "random_state": 42,
        }
        clusterer_params.update(params)
        clusterer = SpectralClustering(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers, center_labels = _median_centers_from_labels(data, labels)

    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        clusterer_params = {
            "n_clusters": n_clusters,
            "linkage": "ward",
        }
        clusterer_params.update(params)
        clusterer = AgglomerativeClustering(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers, center_labels = _median_centers_from_labels(data, labels)

    elif method == "optics":
        from sklearn.cluster import OPTICS

        clusterer_params = {
            "min_samples": 5,
            "xi": 0.05,
            "cluster_method": "xi",
            "metric": metric,
        }
        clusterer_params.update(params)
        clusterer = OPTICS(**clusterer_params)
        labels = clusterer.fit_predict(data)
        centers, center_labels = _median_centers_from_labels(data, labels)

    else:
        supported = [
            "kmeans",
            "dbscan",
            "hdbscan",
            "gmm",
            "spectral",
            "agglomerative",
            "optics",
        ]
        raise ValueError(
            f"Unsupported clustering method: {method}. Supported methods: {supported}"
        )

    return labels, centers, center_labels, clusterer


def cluster_highdimensional_embeddings(
    data,
    method="kmeans",
    n_clusters=8,
    metric="euclidean",
    standardize=False,
    clustering_params=None,
    return_data=False,
    **params,
):
    """
    Cluster an embedding matrix in its original feature space.

    This function is intentionally dimension-agnostic: pass the inference
    embeddings, not the UMAP projection, when you want high-dimensional
    clustering.
    """
    clustering_params = {} if clustering_params is None else dict(clustering_params)
    clustering_params.update(params)
    prepared_data, scaler = _prepare_metric_data(data, standardize=standardize)

    labels, centers, center_labels, clusterer = _fit_clustering(
        prepared_data,
        method=method,
        n_clusters=n_clusters,
        metric=metric,
        params=clustering_params,
    )
    metrics = _cluster_quality_metrics(
        prepared_data,
        labels,
        metric=metric,
        clusterer=clusterer,
    )
    metrics["method"] = method.lower()
    metrics["metric"] = metric
    metrics["standardized"] = bool(standardize)

    result = {
        "labels": labels,
        "centers_highdim": centers,
        "center_labels": center_labels,
        "metrics": metrics,
        "clusterer": clusterer,
        "scaler": scaler,
    }
    if return_data:
        result["clustering_data"] = prepared_data

    return result


def apply_clustering(
    data,
    method="kmeans",
    n_clusters=8,
    metric="euclidean",
    standardize=False,
    clustering_params=None,
    **params,
):
    """
    Notebook-compatible clustering dispatcher.

    Returns `(labels, centers, metrics)` for the requested method.
    """
    combined_params = {} if clustering_params is None else dict(clustering_params)
    combined_params.update(params)
    result = cluster_highdimensional_embeddings(
        data,
        method=method,
        n_clusters=n_clusters,
        metric=metric,
        standardize=standardize,
        clustering_params=combined_params,
    )
    return result["labels"], result["centers_highdim"], result["metrics"]


def cluster_highdimensional_results(
    inference_dir,
    method="kmeans",
    n_clusters=8,
    metric="euclidean",
    standardize=False,
    clustering_params=None,
    **params,
):
    """Load a Hyrax inference directory and cluster its high-dimensional tensors."""
    highdim_data, object_ids = load_result_tensors(inference_dir, flatten=True)
    result = cluster_highdimensional_embeddings(
        highdim_data,
        method=method,
        n_clusters=n_clusters,
        metric=metric,
        standardize=standardize,
        clustering_params=clustering_params,
        **params,
    )
    result["ids"] = object_ids
    return result


def _project_centers_to_umap(centers, clustering_data, umap_points, metric):
    if centers is None:
        return None

    from sklearn.metrics import pairwise_distances

    projected = []
    for center in centers:
        distances = pairwise_distances(
            np.asarray(center).reshape(1, -1),
            clustering_data,
            metric=metric,
        )[0]
        projected.append(umap_points[int(np.argmin(distances)), :2])

    return np.asarray(projected)


def plot_highdim_clusters_on_umap(
    ax,
    config=None,
    inference_dir=None,
    umap_dir=None,
    clustering_method="kmeans",
    n_clusters=8,
    clustering_params=None,
    metric="euclidean",
    standardize=False,
    align_on_ids=True,
    missing="raise",
    alpha=0.6,
    s=5,
    cmap="tab10",
    show_cluster_centers=False,
    show_metrics=True,
    title=None,
    supress_hyrax_logs=False,
):
    """
    Cluster in high-dimensional inference space and visualize labels on UMAP.

    `config` and `supress_hyrax_logs` are accepted for compatibility with the
    notebook prototype; the implementation reads Hyrax batch files directly.
    """
    del config, supress_hyrax_logs

    if inference_dir is None:
        raise ValueError("inference_dir parameter is required")
    if umap_dir is None:
        raise ValueError("umap_dir parameter is required")

    highdim_data, highdim_ids = load_result_tensors(inference_dir, flatten=True)
    umap_points, umap_ids = load_result_tensors(umap_dir, flatten=True)

    if umap_points.shape[1] < 2:
        raise ValueError(
            f"UMAP results must have at least two columns, got {umap_points.shape[1]}"
        )

    if align_on_ids:
        highdim_data, umap_points, aligned_ids = align_tensors_by_id(
            highdim_data,
            highdim_ids,
            umap_points,
            umap_ids,
            missing=missing,
        )
    else:
        if len(highdim_data) != len(umap_points):
            raise ValueError(
                "High-dimensional and UMAP arrays have different lengths: "
                f"{len(highdim_data)} vs {len(umap_points)}"
            )
        aligned_ids = highdim_ids

    result = cluster_highdimensional_embeddings(
        highdim_data,
        method=clustering_method,
        n_clusters=n_clusters,
        metric=metric,
        standardize=standardize,
        clustering_params=clustering_params,
        return_data=True,
    )
    cluster_labels = result["labels"]
    centers_2d = _project_centers_to_umap(
        result["centers_highdim"],
        result["clustering_data"],
        umap_points,
        metric=metric,
    )

    scatter = ax.scatter(
        umap_points[:, 0],
        umap_points[:, 1],
        c=cluster_labels,
        alpha=alpha,
        s=s,
        cmap=cmap,
    )

    if show_cluster_centers and centers_2d is not None:
        ax.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c="red",
            marker="x",
            s=100,
            linewidths=3,
            label="Centers (nearest UMAP points)",
        )
        ax.legend()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    metrics = result["metrics"]
    if title is None:
        title_parts = [f"{clustering_method.upper()} (High-Dim)"]
        if show_metrics:
            if "silhouette_score" in metrics:
                title_parts.append(f"Silhouette: {metrics['silhouette_score']:.3f}")
            title_parts.append(f"Clusters: {metrics['n_clusters']}")
        title = " | ".join(title_parts)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    return {
        "ax": ax,
        "labels": cluster_labels,
        "ids": aligned_ids,
        "centers_highdim": result["centers_highdim"],
        "center_labels": result["center_labels"],
        "centers_2d": centers_2d,
        "metrics": metrics,
        "clusterer": result["clusterer"],
        "scaler": result["scaler"],
    }


def _valid_label_mask(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in {"f", "c"}:
        return np.isfinite(labels)
    return np.asarray(
        [label is not None and str(label).lower() != "nan" for label in labels],
        dtype=bool,
    )


def _sample_group_distances(
    data,
    group_a,
    group_b=None,
    metric="euclidean",
    max_pairs=100000,
    rng=None,
):
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics.pairwise import paired_distances

    rng = np.random.default_rng() if rng is None else rng
    group_a = np.asarray(group_a)

    if group_b is None:
        if len(group_a) < 2:
            return np.array([])
        n_pairs = len(group_a) * (len(group_a) - 1) // 2
        if n_pairs <= max_pairs:
            distances = pairwise_distances(data[group_a], metric=metric)
            return distances[np.triu_indices(len(group_a), k=1)]

        left = rng.choice(group_a, size=max_pairs * 2, replace=True)
        right = rng.choice(group_a, size=max_pairs * 2, replace=True)
        keep = left != right
        left = left[keep][:max_pairs]
        right = right[keep][:max_pairs]
        if len(left) == 0:
            return np.array([])
        return paired_distances(data[left], data[right], metric=metric)

    group_b = np.asarray(group_b)
    if len(group_a) == 0 or len(group_b) == 0:
        return np.array([])

    n_pairs = len(group_a) * len(group_b)
    if n_pairs <= max_pairs:
        return pairwise_distances(data[group_a], data[group_b], metric=metric).ravel()

    left = rng.choice(group_a, size=max_pairs, replace=True)
    right = rng.choice(group_b, size=max_pairs, replace=True)
    return paired_distances(data[left], data[right], metric=metric)


def _add_distance_summary(metrics, prefix, distances):
    distances = np.asarray(distances)
    metrics[f"{prefix}_n_pairs"] = int(len(distances))
    if len(distances) == 0:
        metrics[f"{prefix}_distance_mean"] = np.nan
        metrics[f"{prefix}_distance_median"] = np.nan
        return

    metrics[f"{prefix}_distance_mean"] = float(np.mean(distances))
    metrics[f"{prefix}_distance_median"] = float(np.median(distances))


def evaluate_labeled_distances(
    data,
    labels,
    positive_label=None,
    metric="euclidean",
    standardize=False,
    n_neighbors=15,
    max_distance_pairs=100000,
    random_state=42,
):
    """
    Quantify whether labeled objects cluster in high-dimensional space.

    For merger work, pass high-dimensional inference embeddings as `data`,
    a merger flag/time-window label array as `labels`, and the merger value as
    `positive_label` (for example True or 1).
    """
    data, _ = _prepare_metric_data(data, standardize=standardize)
    labels = np.asarray(labels)

    if len(labels) != len(data):
        raise ValueError(
            f"labels length must match data length: {len(labels)} vs {len(data)}"
        )

    valid = _valid_label_mask(labels)
    data = data[valid]
    labels = labels[valid]

    if len(labels) < 2:
        raise ValueError("At least two labeled samples are required")

    unique_labels, counts = np.unique(labels, return_counts=True)
    metrics = {
        "n_samples": int(len(labels)),
        "n_labels": int(len(unique_labels)),
        "label_counts": {
            str(label): int(count) for label, count in zip(unique_labels, counts)
        },
        "metric": metric,
        "standardized": bool(standardize),
    }

    if len(unique_labels) > 1 and len(labels) > len(unique_labels):
        from sklearn.metrics import silhouette_score

        metrics["silhouette_score"] = float(
            silhouette_score(data, labels, metric=metric)
        )

    if n_neighbors is not None and len(labels) > 1:
        from sklearn.neighbors import NearestNeighbors

        n_neighbors_used = min(int(n_neighbors) + 1, len(labels))
        if n_neighbors_used > 1:
            neighbor_model = NearestNeighbors(
                n_neighbors=n_neighbors_used,
                metric=metric,
            )
            neighbor_indexes = neighbor_model.fit(data).kneighbors(
                return_distance=False
            )[:, 1:]
            same_label = labels[neighbor_indexes] == labels[:, None]
            metrics["n_neighbors_used"] = int(neighbor_indexes.shape[1])
            metrics["mean_neighbor_same_label_fraction"] = float(
                np.mean(same_label)
            )

    if positive_label is not None:
        positive_mask = labels == positive_label
        if not np.any(positive_mask):
            positive_mask = labels.astype(str) == str(positive_label)

        positive_indexes = np.flatnonzero(positive_mask)
        other_indexes = np.flatnonzero(~positive_mask)
        metrics["positive_label"] = str(positive_label)
        metrics["positive_count"] = int(len(positive_indexes))
        metrics["other_count"] = int(len(other_indexes))
        metrics["positive_fraction"] = float(len(positive_indexes) / len(labels))

        if "n_neighbors_used" in metrics and len(positive_indexes) > 0:
            positive_neighbor_fraction = np.mean(same_label[positive_mask])
            baseline = (
                (len(positive_indexes) - 1) / (len(labels) - 1)
                if len(labels) > 1
                else np.nan
            )
            metrics["positive_neighbor_same_label_fraction"] = float(
                positive_neighbor_fraction
            )
            metrics["positive_neighbor_baseline_fraction"] = float(baseline)
            metrics["positive_neighbor_enrichment"] = float(
                positive_neighbor_fraction / baseline
            ) if baseline > 0 else np.nan

        rng = np.random.default_rng(random_state)
        positive_intra = _sample_group_distances(
            data,
            positive_indexes,
            metric=metric,
            max_pairs=max_distance_pairs,
            rng=rng,
        )
        other_intra = _sample_group_distances(
            data,
            other_indexes,
            metric=metric,
            max_pairs=max_distance_pairs,
            rng=rng,
        )
        positive_other = _sample_group_distances(
            data,
            positive_indexes,
            other_indexes,
            metric=metric,
            max_pairs=max_distance_pairs,
            rng=rng,
        )

        _add_distance_summary(metrics, "positive_intra", positive_intra)
        _add_distance_summary(metrics, "other_intra", other_intra)
        _add_distance_summary(metrics, "positive_other", positive_other)

        if (
            np.isfinite(metrics["positive_intra_distance_mean"])
            and np.isfinite(metrics["positive_other_distance_mean"])
            and metrics["positive_other_distance_mean"] != 0
        ):
            metrics["positive_intra_to_other_distance_ratio"] = float(
                metrics["positive_intra_distance_mean"]
                / metrics["positive_other_distance_mean"]
            )

    return metrics


def _labels_aligned_to_result_ids(labels, result_ids, label_ids=None):
    result_ids = _normalize_result_ids(result_ids)

    if result_ids is None:
        labels = np.asarray(labels)
        return labels

    if label_ids is not None:
        label_ids = _normalize_result_ids(label_ids)
        labels = np.asarray(labels)
        if len(label_ids) != len(labels):
            raise ValueError("label_ids and labels must have the same length")
        label_lookup = {
            object_id: labels[index] for index, object_id in enumerate(label_ids)
        }
    elif isinstance(labels, Mapping):
        label_lookup = {str(key): value for key, value in labels.items()}
    elif hasattr(labels, "to_dict"):
        candidate_lookup = {
            str(key): value for key, value in labels.to_dict().items()
        }
        if any(object_id in candidate_lookup for object_id in result_ids):
            label_lookup = candidate_lookup
        else:
            labels = np.asarray(labels)
            if len(labels) != len(result_ids):
                raise ValueError(
                    "Series-like labels must either be indexed by result ID or "
                    "have the same length as the result tensors"
                )
            return labels
    else:
        labels = np.asarray(labels)
        if len(labels) != len(result_ids):
            raise ValueError(
                "labels must be the same length as the result tensors unless "
                "label_ids or a mapping keyed by object ID is provided"
            )
        return labels

    return np.asarray([label_lookup.get(object_id, np.nan) for object_id in result_ids])


def evaluate_highdim_merger_clustering(
    inference_dir,
    labels,
    label_ids=None,
    positive_label=True,
    metric="euclidean",
    standardize=False,
    n_neighbors=15,
    max_distance_pairs=100000,
    random_state=42,
):
    """
    Load a Hyrax inference directory and evaluate labeled merger clustering.

    `labels` may be an array in inference-result order, a mapping keyed by
    object ID, or an array paired with `label_ids`.
    """
    highdim_data, result_ids = load_result_tensors(inference_dir, flatten=True)
    aligned_labels = _labels_aligned_to_result_ids(
        labels,
        result_ids,
        label_ids=label_ids,
    )
    metrics = evaluate_labeled_distances(
        highdim_data,
        aligned_labels,
        positive_label=positive_label,
        metric=metric,
        standardize=standardize,
        n_neighbors=n_neighbors,
        max_distance_pairs=max_distance_pairs,
        random_state=random_state,
    )
    metrics["inference_dir"] = str(inference_dir)
    return metrics
