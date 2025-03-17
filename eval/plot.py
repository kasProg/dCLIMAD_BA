import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import KNN

def plot_violin_bias(bias_data, bias_label, title, remove_outlier=False):
    if isinstance(bias_data, dict):
        bias_values, bias_types = [], []
        for label, (bias_raw, bias_corrected) in bias_data.items():
            for bias, sublabel in [(bias_raw, "Raw"), (bias_corrected, "Corrected")]:
                if isinstance(bias, np.ndarray) and bias.size > 0:
                    bias_values.extend(bias.tolist())
                    bias_types.extend([f"{label} ({sublabel})"] * len(bias))
        
        df_bias = pd.DataFrame({bias_label: bias_values, "Type": bias_types})

    else:
        bias_raw, bias_corrected = bias_data
        df_bias = pd.DataFrame({bias_label: np.concatenate([bias_raw, bias_corrected]),
        "Type": (["Mean Bias (Raw)"] * len(bias_raw)) + (["Mean Bias (Corrected)"] * len(bias_corrected))})
        
    
    if remove_outlier:
        df_bias = df_bias[df_bias[bias_label]<100]
        df_bias = df_bias[df_bias[bias_label]>-100]

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Type", y=bias_label, data=df_bias, inner="point", cut=0)
    plt.title(title)
    plt.ylabel(bias_label)
    plt.xlabel("Categories")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_spatial_bias(valid_coords, mean_bias_values_array, threshold_type, label, vmin=None, vmax=None):
    lats, lons = zip(*valid_coords)  # Extract latitudes and longitudes
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    if threshold_type==None:
        fig.suptitle("Overall", fontsize=16, fontweight='bold')

        for ax, mean_bias_values, title in zip(axes, [mean_bias_values_array[0], mean_bias_values_array[1]], ['Raw', 'Bias Adjusted']):
            m = Basemap(projection='merc',
                        llcrnrlat=min(lats)-1, urcrnrlat=max(lats)+1,
                        llcrnrlon=min(lons)-1, urcrnrlon=max(lons)+1,
                        resolution='i', ax=ax)
            m.drawcoastlines()
            m.drawcountries()
            x, y = m(lons, lats)
            sc = m.scatter(x, y, c=mean_bias_values, cmap="coolwarm", marker="o", edgecolor="k", alpha=0.75, vmin=vmin, vmax=vmax)
            
            ax.set_title(title)
    else:

        fig.suptitle(threshold_type, fontsize=16, fontweight='bold')

        for ax, mean_bias_values, title in zip(axes, [mean_bias_values_array[threshold_type][0], mean_bias_values_array[threshold_type][1]], ['Raw', 'Bias Adjusted']):
            m = Basemap(projection='merc',
                        llcrnrlat=min(lats)-1, urcrnrlat=max(lats)+1,
                        llcrnrlon=min(lons)-1, urcrnrlon=max(lons)+1,
                        resolution='i', ax=ax)
            m.drawcoastlines()
            m.drawcountries()
            # m.drawparallels(np.arange(int(min(lats)), int(max(lats)), 2), labels=[1,0,0,0], fontsize=10)
            # m.drawmeridians(np.arange(int(min(lons)), int(max(lons)), 2), labels=[0,0,0,1], fontsize=10)
            
            x, y = m(lons, lats)
            sc = m.scatter(x, y, c=mean_bias_values, cmap="coolwarm", marker="o", edgecolor="k", alpha=0.75, vmin=vmin, vmax=vmax)
            
            ax.set_title(title)
            
    
    fig.colorbar(sc, ax=axes, orientation="vertical", fraction=0.02, pad=0.05, label=label)
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