from pathlib import Path
import re
import numpy as np
from pathlib import Path
import logging
import hyrax


def extract_umap_info(x,y,base_directory="/work/hdd/bemi/dmiura/800batch/test_run"):
  """Extract UMAP directory from 3dumap output file."""

  # Construct paths
  run_dir = Path(base_directory) # / f"run{x}"
  dumap_name = f"udb{x}_{y}"

  dumap_output_file = run_dir / f"{dumap_name}.err"  # .txt"
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


import matplotlib
matplotlib.use('Agg')  # before importing pyplot
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Call your function
plot_umap_simple(
    ax,
    config=extract_umap_info(1,1)[1],                # replace with your Hyrax config
    input_dir=extract_umap_info(1,1)[0],   # path returned from extract_umap_info
    alpha=0.8,
    s=1, 
    cmap='plasma', 
    density=True,
    supress_hyrax_logs=False, 
    log_colorbar=True  # optional
)

# Show the figure (if you want it to pop up)
# plt.show()

# OR save to file if running in a headless terminal (e.g. on HPC)
fig.savefig("umap_plot.png", dpi=300)
