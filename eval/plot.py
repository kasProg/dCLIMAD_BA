import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import KNN

def plot_violin_bias(ax, bias_data, bias_label, title, method_names=None, remove_outlier=False):
    """
    Function to plot a violin plot comparing raw vs. multiple corrected bias data with method names.

    Parameters:
        ax (matplotlib.axes): Subplot axis to plot on.
        bias_data (tuple or dict): 
            - If tuple: (raw, corrected_1, corrected_2, ...) for a single dataset.
            - If dict: {"Category 1": (raw, corrected_1, corrected_2, ...), "Category 2": ...}
        bias_label (str): Label for the y-axis.
        title (str): Title of the plot.
        method_names (list, optional): Names for the bias correction methods (e.g., ["Quantile Mapping", "Neural Network"]).
        remove_outlier (bool): If True, filters extreme values beyond ±100.
    """
    bias_values, bias_types, categories = [], [], []

    if isinstance(bias_data, dict):  
        # Handling multiple categories (e.g., different precipitation classes)
        for category, biases in bias_data.items():
            bias_raw, *bias_corrected_methods = biases
            
            # Store raw data
            if isinstance(bias_raw, np.ndarray) and bias_raw.size > 0:
                bias_values.extend(bias_raw.tolist())
                bias_types.extend(["Raw"] * len(bias_raw))
                categories.extend([category] * len(bias_raw))
            
            # Store each correction method dynamically
            for i, bias_corrected in enumerate(bias_corrected_methods):
                method_label = method_names[i] if method_names and i < len(method_names) else f"Method {i+1}"
                if isinstance(bias_corrected, np.ndarray) and bias_corrected.size > 0:
                    bias_values.extend(bias_corrected.tolist())
                    bias_types.extend([method_label] * len(bias_corrected))
                    categories.extend([category] * len(bias_corrected))

    else:  
        # Single dataset with multiple correction methods
        bias_raw, *bias_corrected_methods = bias_data

        if isinstance(bias_raw, np.ndarray) and bias_raw.size > 0:
            bias_values.extend(bias_raw.tolist())
            bias_types.extend(["Raw"] * len(bias_raw))
            categories.extend(["All Data"] * len(bias_raw))

        for i, bias_corrected in enumerate(bias_corrected_methods):
            method_label = method_names[i] if method_names and i < len(method_names) else f"Method {i+1}"
            if isinstance(bias_corrected, np.ndarray) and bias_corrected.size > 0:
                bias_values.extend(bias_corrected.tolist())
                bias_types.extend([method_label] * len(bias_corrected))
                categories.extend(["All Data"] * len(bias_corrected))

    df_bias = pd.DataFrame({bias_label: bias_values, "Method": bias_types, "Category": categories})


    if remove_outlier:
        df_bias = df_bias[(df_bias[bias_label] < 100) & (df_bias[bias_label] > -100)]

    # Define colors: Raw = Blue, Corrected Methods = Different Colors
    # Define consistent color mapping: Assign colors dynamically
    unique_methods = ["Raw"] + (method_names if method_names else [f"Method {i+1}" for i in range(len(df_bias["Method"].unique()) - 1)])
    color_palette = sns.color_palette("husl", len(unique_methods))

    # Map method names to colors
    color_mapping = {method: color_palette[i] for i, method in enumerate(unique_methods)}

    # Apply colors in orderx
    palette = [color_mapping[method] for method in df_bias["Method"].unique()]
    # unique_types = df_bias["Type"].unique()
    # palette = ["C0" if "Raw" in x else f"C{i+1}" for i, x in enumerate(unique_types)]

    # sns.violinplot(x="Type", y=bias_label, data=df_bias, inner="point", cut=0, palette=palette, ax=ax)
    sns.violinplot(x="Category", y=bias_label, hue="Method", data=df_bias, inner="point", cut=0, palette=color_mapping, ax=ax)
    
     # Draw zero line
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Improve legend placement
    ax.legend(title="Adjustment Method", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Format x-axis to create hierarchical labels
    category_positions = {category: i for i, category in enumerate(df_bias["Category"].unique())}

    # Set tick labels: Show category at bottom, methods above
    ax.set_xticks(list(category_positions.values()))
    ax.set_xticklabels(list(category_positions.keys()), fontsize=12, fontweight='bold')

    ax.set_xlabel("Precipitation Index")
    ax.set_title(title)
    ax.set_ylabel(bias_label)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


# def plot_spatial_bias(valid_coords, mean_bias_values_array, threshold_type, label, vmin=None, vmax=None):
#     lats, lons = zip(*valid_coords)  # Extract latitudes and longitudes
    
#     fig, axes = plt.subplots(1, 2, figsize=(16, 4))

#     if threshold_type==None:
#         fig.suptitle("Overall", fontsize=16, fontweight='bold')

#         for ax, mean_bias_values, title in zip(axes, [mean_bias_values_array[0], mean_bias_values_array[1]], ['Raw', 'Bias Adjusted']):
#             m = Basemap(projection='merc',
#                         llcrnrlat=min(lats)-1, urcrnrlat=max(lats)+1,
#                         llcrnrlon=min(lons)-1, urcrnrlon=max(lons)+1,
#                         resolution='i', ax=ax)
#             m.drawcoastlines()
#             m.drawcountries()
#             x, y = m(lons, lats)
#             sc = m.scatter(x, y, c=mean_bias_values, cmap="coolwarm", marker="o", edgecolor="k", alpha=0.75, vmin=vmin, vmax=vmax)
            
#             ax.set_title(title)
#     else:

#         fig.suptitle(threshold_type, fontsize=16, fontweight='bold')

#         for ax, mean_bias_values, title in zip(axes, [mean_bias_values_array[threshold_type][0], mean_bias_values_array[threshold_type][1]], ['Raw', 'Bias Adjusted']):
#             m = Basemap(projection='merc',
#                         llcrnrlat=min(lats)-1, urcrnrlat=max(lats)+1,
#                         llcrnrlon=min(lons)-1, urcrnrlon=max(lons)+1,
#                         resolution='i', ax=ax)
#             m.drawcoastlines()
#             m.drawcountries()
#             # m.drawparallels(np.arange(int(min(lats)), int(max(lats)), 2), labels=[1,0,0,0], fontsize=10)
#             # m.drawmeridians(np.arange(int(min(lons)), int(max(lons)), 2), labels=[0,0,0,1], fontsize=10)
            
#             x, y = m(lons, lats)
#             sc = m.scatter(x, y, c=mean_bias_values, cmap="coolwarm", marker="o", edgecolor="k", alpha=0.75, vmin=vmin, vmax=vmax)
            
#             ax.set_title(title)
            
    
#     fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.05, label=label)
#     plt.show()

def plot_spatial_bias(valid_coords, mean_bias_values_array, threshold_type, label, 
                              method_names=None, vmin=None, vmax=None, fig=None, axes=None):
    """
    Function to plot spatial maps of bias for Raw and multiple bias correction methods.
    Supports both standalone plots and subplots.

    Parameters:
        valid_coords (list of tuples): List of (lat, lon) coordinates.
        mean_bias_values_array (tuple or dict): 
            - If tuple: (raw, corrected_1, corrected_2, ...).
            - If dict: {"Category 1": (raw, corrected_1, corrected_2, ...), "Category 2": ...}.
        threshold_type (str or None): Category name for the title (e.g., "Low Rain"). If None, uses "Overall".
        label (str): Colorbar label.
        method_names (list, optional): Names of bias correction methods (e.g., ["Quantile Mapping", "Neural Network"]).
        vmin, vmax (float, optional): Minimum and maximum color limits for color consistency.
        fig (matplotlib.figure.Figure, optional): External figure object for subplots.
        axes (array of matplotlib.axes.Axes, optional): External axes array for subplots.
    """

    lats, lons = zip(*valid_coords)  # Extract latitudes and longitudes

    if isinstance(mean_bias_values_array, dict):
        mean_bias_values_list = mean_bias_values_array[threshold_type]  # Extract specific threshold values
    else:
        mean_bias_values_list = mean_bias_values_array  # Use the provided tuple

    # Determine the number of methods dynamically
    num_methods = len(mean_bias_values_list)  # Includes Raw + Methods
    method_labels = ["Raw"] + (method_names if method_names else [f"Method {i+1}" for i in range(num_methods - 1)])

    # Create a figure & axes only if they are not provided (standalone mode)
    if axes is None:
        fig, axes = plt.subplots(1, num_methods, figsize=(6 * num_methods, 5), sharex=True, sharey=True)

    # Convert to an array if a single axis is passed
    if num_methods == 1:
        axes = [axes]

    # Set a super title only if this is a standalone plot
    if fig is not None:
        fig.suptitle(threshold_type if threshold_type else "Overall", fontsize=16, fontweight="bold")
    
    # Keep track of the last scatter plot for colorbar reference
    last_scatter = None

    for ax, mean_bias_values, title in zip(axes, mean_bias_values_list, method_labels):
        # Initialize Basemap for each subplot
        m = Basemap(projection="merc",
                    llcrnrlat=min(lats) - 1, urcrnrlat=max(lats) + 1,
                    llcrnrlon=min(lons) - 1, urcrnrlon=max(lons) + 1,
                    resolution="i", ax=ax)

        m.drawcoastlines()
        m.drawcountries()

        x, y = m(lons, lats)
        last_scatter = m.scatter(x, y, c=mean_bias_values, cmap="coolwarm", marker="o", edgecolor="k", alpha=0.75, vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=12, fontweight="bold")

    # Add colorbar only if this is a standalone figure
    if fig is not None and last_scatter is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # Manually position colorbar
        cbar = fig.colorbar(last_scatter, cax=cbar_ax)
        cbar.set_label(label, fontsize=12)

    # Adjust layout if this is a standalone figure
    if fig is not None:
        # plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
        plt.show()

# Function to plot Moran Scatter Plot
def plot_moran_scatter(values, valid_coords, k=5):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon for lat, lon in valid_coords], 
                                                        [lat for lat, lon in valid_coords]))
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = 'R'
    moran = Moran(values, w)
    
    # Compute spatial lag
    spatial_lag = np.dot(w.full()[0], values)
    
    # Plot Moran scatter plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=values, y=spatial_lag, scatter_kws={"s": 30}, line_kws={"color": "red"})
    plt.xlabel("Bias Value")
    plt.ylabel("Spatially Lagged Bias Value")
    plt.title(f"Moran Scatter Plot (Moran's I: {moran.I:.4f}, p={moran.p_sim:.3f})")
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()