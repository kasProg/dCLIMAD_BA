
import pandas as pd
import numpy as np
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import KNN

# Function to determine season from month
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring" #MAM
    elif month in [6, 7, 8]:
        return "Summer" #JJA
    elif month in [9, 10, 11]:
        return "Autumn" #SON
    elif month in [12, 1, 2]:
        return "Winter" #DJF
    

def load_seasonal_data(time_series, data):
    months = pd.to_datetime(time_series).month  # Extract months from time series
    seasons = np.array([get_season(m) for m in months])  # Assign seasons
    
    seasonal_data = {
        "Spring": data[seasons == "Spring"],
        "Summer": data[seasons == "Summer"],
        "Autumn": data[seasons == "Autumn"],
        "Winter": data[seasons == "Winter"]
    }
    
    return seasonal_data

# Function to compute Rx1day (monthly max 1-day precipitation)
def compute_rx1day(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    return df.resample('ME').max().values  # Monthly max precipitation

# Function to compute Rx5day (monthly max 5-day precipitation)
def compute_rx5day(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    rolling_sum = df.rolling(window=5, min_periods=1).sum()  # 5-day rolling sum
    return rolling_sum.resample('ME').max().values  # Monthly max 5-day precipitation

# Function to compute SDII (Simple Precipitation Intensity Index)
def compute_sdii(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    wet_days = df[df >= 1]  # Filter only wet days (>=1mm)
    return (wet_days.resample('ME').sum() / wet_days.resample('ME').count()).fillna(0).values  # Monthly SDII

# Function to compute R10mm (Annual count of days with PRCP ≥ 10mm)
def compute_r10mm(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    return (df >= 10).resample('YE').sum().values  # Annual count of days with PRCP ≥ 10mm

# Function to compute R20mm (Annual count of days with PRCP ≥ 20mm)
def compute_r20mm(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    return (df >= 20).resample('YE').sum().values  # Annual count of days with PRCP ≥ 20mm

# # Function to compute CDD (Consecutive Dry Days)
# def compute_cdd(time, y):
#     df = pd.DataFrame(y < 1, index=pd.to_datetime(time))  # Mask for dry days (PRCP < 1mm)
#     return df.resample('YE').apply(lambda x: x.astype(int).groupby((x != x.shift()).cumsum()).sum().max()).values

def get_longest_streak(time, y, threshold=1.0, condition='lt'):
    """
    Compute the longest consecutive run of days per year based on a threshold.

    Parameters:
        time (array-like): Timestamps for each day.
        y (array-like): Daily values (e.g., precipitation).
        threshold (float): Threshold for dry/wet day cutoff (e.g., 1mm).
        condition (str): 'lt' for < threshold (dry), 'ge' for ≥ threshold (wet).

    Returns:
        np.ndarray: Array of longest consecutive streaks per year.
    """
    time = pd.to_datetime(time)
    if condition == 'lt':
        mask = (y < threshold)
    elif condition == 'ge':
        mask = (y >= threshold)
    else:
        raise ValueError("condition must be 'lt' or 'ge'")

    df = pd.DataFrame(mask, index=time)

    def longest_run(series):
        s = series.astype(int)
        return s.groupby((s != s.shift()).cumsum()).sum().max()

    result = df.resample('YE').apply(lambda x: longest_run(x))
    return result.to_numpy()


def compute_cdd(time, y):
    """Compute Consecutive Dry Days (CDD): PRCP < 1mm"""
    return get_longest_streak(time, y, threshold=1.0, condition='lt')

def compute_cwd(time, y):
    """Compute Consecutive Wet Days (CWD): PRCP ≥ 1mm"""
    return get_longest_streak(time, y, threshold=1.0, condition='ge')


# # Function to compute CWD (Consecutive Wet Days)
# def compute_cwd(time, y):
#     df = pd.DataFrame(y >= 1, index=pd.to_datetime(time))  # Mask for wet days (PRCP ≥ 1mm)
#     return df.resample('YE').apply(lambda x: x.astype(int).groupby((x != x.shift()).cumsum()).sum().max()).values  # Max consecutive wet days



# Function to compute R95pTOT (Annual total precipitation above 95th percentile)
def compute_r95ptot(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    threshold = df[df >= 1].quantile(0.95)  # 95th percentile of wet days
    return df[df > threshold].resample('YE').sum().values  # Sum of precipitation above 95th percentile

# Function to compute R99pTOT (Annual total precipitation above 99th percentile)
def compute_r99ptot(time, y):
    df = pd.DataFrame(y, index=pd.to_datetime(time))
    threshold = df[df >= 1].quantile(0.99)  # 99th percentile of wet days
    return df[df > threshold].resample('YE').sum().values 

# Generalized threshold manager class
class ClimateIndices:
    def __init__(self):
        self.indices = {
            "Dry Days": (1, np.less_equal),
            "Wet Days >1mm": (1, np.greater),
            "Very Wet Days >10mm": (10, np.greater),
            "Very Very Wet Days >20mm": (20, np.greater),
            "Rx1day": (compute_rx1day, None),
            "Rx5day": (compute_rx5day, None),
            "SDII (Monthly)": (compute_sdii, None),
            "R10mm": (compute_r10mm, None),
            "R20mm": (compute_r20mm, None),
            "CDD (Yearly)": (compute_cdd, None),
            "CWD (Yearly)": (compute_cwd, None),
            "R95pTOT": (compute_r95ptot, None),
            "R99pTOT": (compute_r95ptot, None),
        }
    
    def add_index(self, name, threshold, comparison=None):
        """ Adds a new climate index with a threshold and comparison function """
        self.indices[name] = (threshold, comparison)
    
    def get_indices(self):
        return self.indices


# Function to compute mean bias per coordinate
def compute_mean_bias(mask, x, y, xt):
    if mask is None:
        bias_x = np.nanmean(x - y, axis=0)
        bias_xt = np.nanmean(xt - y, axis=0)
    else:
        mask_expanded = np.where(mask, 1, np.nan)  # Convert mask to NaN-based filter
        bias_x = np.nanmean((x - y) * mask_expanded, axis=0)  # Compute mean bias location-wise
        bias_xt = np.nanmean((xt - y) * mask_expanded, axis=0)
    return bias_x, bias_xt

def compute_mean_bias_percentage(mask, x, y, xt):
    if mask is None:
        percent_bias_x = np.nanmean(((x - y) / (y + 1e-6)) * 100, axis=0)
        percent_bias_xt = np.nanmean(((xt - y) / (y + 1e-6)) * 100, axis=0)
    else:
        mask_expanded = np.where(mask, 1, np.nan)  # Convert mask to NaN-based filter
        percent_bias_x = np.nanmean(((x - y) / (y + 1e-6)) * 100 * mask_expanded, axis=0)  # Compute mean percentage bias location-wise
        percent_bias_xt = np.nanmean(((xt - y) / (y + 1e-6)) * 100 * mask_expanded, axis=0)
    return percent_bias_x, percent_bias_xt

# Function to compute bias in the number of days per category
def compute_day_bias(threshold, x, y, xt, comparison):
    count_x = np.sum(comparison(x, threshold), axis=0)  # Count occurrences in raw data
    count_xt = np.sum(comparison(xt, threshold), axis=0)  # Count occurrences in corrected data
    count_y = np.sum(comparison(y, threshold), axis=0)  # Reference count
    return count_x - count_y, count_xt - count_y

# Function to compute percentage bias in the number of days per category
def compute_day_bias_percentage(threshold, x, y, xt, comparison):
    count_x = np.sum(comparison(x, threshold), axis=0)
    count_xt = np.sum(comparison(xt, threshold), axis=0)
    count_y = np.sum(comparison(y, threshold), axis=0)
    percent_x = ((count_x - count_y) / (count_y + 1e-6)) * 100  # Avoid division by zero
    percent_xt = ((count_xt - count_y) / (count_y + 1e-6)) * 100
    return percent_x, percent_xt

def compute_mae(mask, x, y, xt):
    if mask is None:
        mae_x = np.nanmean(np.abs(x - y), axis=0)
        mae_xt = np.nanmean(np.abs(xt - y), axis=0)
    else:
        mask_expanded = np.where(mask, 1, np.nan)  # Convert mask to NaN-based filter
        mae_x = np.nanmean(np.abs(x - y) * mask_expanded, axis=0)
        mae_xt = np.nanmean(np.abs(xt - y) * mask_expanded, axis=0)
    return mae_x, mae_xt

def compute_rmse(mask, x, y, xt):
    if mask is None:
        rmse_x = np.sqrt(np.mean((x - y) ** 2, axis=0))
        rmse_xt = np.sqrt(np.mean((xt - y) ** 2, axis=0))
    else:
        mask_expanded = np.where(mask, 1, np.nan)  # Convert mask to NaN-based filter
        rmse_x = np.sqrt(np.nanmean((x - y) ** 2 * mask_expanded, axis=0))
        rmse_xt = np.sqrt(np.nanmean((xt - y) ** 2 * mask_expanded, axis=0))
    return  rmse_x, rmse_xt

# Generalized function to compute mean bias for different precipitation categories
def get_mean_biases(x, y, xt, time, indices_manager):
    biases = {}
    thresholds = indices_manager.get_indices()
    for label, (threshold, comparison) in thresholds.items():
        if callable(threshold):  # If function, apply it to y
            computed_x = threshold(time, x)
            computed_y = threshold(time, y)
            computed_xt = threshold(time, xt)
            biases[label] = compute_mean_bias(None, computed_x, computed_y, computed_xt)
        else:
            biases[label] = compute_mean_bias(comparison(y, threshold), x, y, xt)
    return biases

# Generalized function to compute mean bias for different precipitation categories
def get_mean_bias_percentages(x, y, xt, time, indices_manager):
    biases = {}
    thresholds = indices_manager.get_indices()
    for label, (threshold, comparison) in thresholds.items():
        if callable(threshold):  # If function, apply it to y
            computed_x = threshold(time, x)
            computed_y = threshold(time, y)
            computed_xt = threshold(time, xt)
            biases[label] = compute_mean_bias_percentage(None, computed_x, computed_y, computed_xt)
        else:
            biases[label] = compute_mean_bias_percentage(comparison(y, threshold), x, y, xt)
    return biases

# Generalized function to compute rmse
def get_rmse(x, y, xt, time, indices_manager):
    rmse = {}
    thresholds = indices_manager.get_indices()
    for label, (threshold, comparison) in thresholds.items():
        if callable(threshold):  # If function, apply it to y
            computed_x = threshold(time, x)
            computed_y = threshold(time, y)
            computed_xt = threshold(time, xt)
            rmse[label] = compute_rmse(None, computed_x, computed_y, computed_xt)
        else:
            rmse[label] = compute_rmse(comparison(y, threshold), x, y, xt)
    return rmse

# Generalized function to compute mae
def get_mae(x, y, xt, time, indices_manager):
    mae = {}
    thresholds = indices_manager.get_indices()
    for label, (threshold, comparison) in thresholds.items():
        if callable(threshold):  # If function, apply it to y
            computed_x = threshold(time, x)
            computed_y = threshold(time, y)
            computed_xt = threshold(time, xt)
            mae[label] = compute_mae(None, computed_x, computed_y, computed_xt)
        else:
            mae[label] = compute_mae(comparison(y, threshold), x, y, xt)
    return mae



# Generalized function to compute day bias for different precipitation categories
def get_day_biases(x, y, xt, indices_manager):
    return {label: compute_day_bias(threshold, x, y, xt, comparison)
            for label, (threshold, comparison) in indices_manager.get_indices().items() if comparison is not None}

# Generalized function to compute day bias for different precipitation categories
def get_day_bias_percentages(x, y, xt, indices_manager):
    return {label: compute_day_bias_percentage(threshold, x, y, xt, comparison)
            for label, (threshold, comparison) in indices_manager.get_indices().items() if comparison is not None}


# Function to compute Moran's I for spatial autocorrelation
def compute_morans_i(values, valid_coords, k=5):
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon for lat, lon in valid_coords], 
                                                        [lat for lat, lon in valid_coords]))
    w = KNN.from_dataframe(gdf, k=k)  # Compute spatial weights using k-nearest neighbors
    w.transform = 'R'  # Row-standardized weights
    moran = Moran(values, w)  # Compute Moran's I
    return moran.I, moran.p_sim  # Return Moran's I value and p-value