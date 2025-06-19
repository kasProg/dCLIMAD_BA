
import pandas as pd
import numpy as np
import geopandas as gpd
from esda.moran import Moran
from libpysal.weights import KNN
import warnings
from scipy.stats import genextreme
from scipy.stats import bootstrap
from scipy.stats import gaussian_kde
warnings.simplefilter(action='ignore', category=FutureWarning)


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

def compute_mean_bias_percentage(mask, x, y, xt, threshold=1.0):
    """
    Compute percentage bias over time at each grid cell.
    x, y, xt: Arrays of shape (time, n_grid)
    mask: Boolean array of shape (n_grid,) or None
    threshold: minimum y value to consider valid
    """
    # Only consider points where y > threshold
    valid = y > threshold  # shape: (time, n_grid)

    if mask is not None:
        mask = mask.astype(bool)
        # Expand mask to match shape (time, n_grid)
        valid = valid & mask

    # Avoid divide-by-zero; set invalid comparisons to nan
    with np.errstate(divide='ignore', invalid='ignore'):
        bias_x = np.where(valid, ((x - y) / y) * 100, np.nan)
        bias_xt = np.where(valid, ((xt - y) / y) * 100, np.nan)

    # Now take mean over time axis (axis=0), preserving spatial dimension
    percent_bias_x = np.nanmean(bias_x, axis=0)  # shape: (n_grid,)
    percent_bias_xt = np.nanmean(bias_xt, axis=0)

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




def calculate_annual_maxima(data, time_arr):
    # Convert time array to pandas datetime for grouping
    time_pd = pd.to_datetime(time_arr)

    # Extract year from each timestamp
    years = time_pd.year

    # Create a DataFrame to associate each timestamp with its year
    df_years = pd.DataFrame({'year': years})

    # Find the unique years
    unique_years = np.unique(years)

    # Preallocate output: (years, coordinates)
    annual_max = np.full((len(unique_years), data.shape[1]), np.nan)

    for i, yr in enumerate(unique_years):
        # Indices where the year matches
        idx = df_years['year'] == yr

        # Max over time axis (i.e., for that year's data)
        annual_max[i] = np.nanmax(data[idx.values], axis=0)

    return annual_max


def fit_stationary_gev(data, return_periods=[5, 10, 20, 50, 100], n_bootstrap=1000, ci=0.95):
    """
    Fits a stationary GEV distribution and computes return level metrics with CI.
    
    Parameters:
        data: 1D array of annual maxima
        return_periods: list of return periods
        n_bootstrap: number of bootstrap samples
        ci: confidence level
    
    Returns:
        dict of return period → (low, best, high) values
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    
    # Fit GEV: scipy uses c=-shape
    shape, loc, scale = genextreme.fit(data)
    
    alpha = 1 - ci
    z = {}

    # Calculate return levels for best-fit GEV
    for rp in return_periods:
        prob = 1 - 1/rp
        rl_best = genextreme.ppf(prob, shape, loc=loc, scale=scale)
        z[rp] = {'best': rl_best, 'low': None, 'high': None}

    # Bootstrap CI
    boot_vals = []

    rng = np.random.default_rng()
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        try:
            s, l, sc = genextreme.fit(sample)
            rl = [genextreme.ppf(1 - 1/rp, s, loc=l, scale=sc) for rp in return_periods]
            boot_vals.append(rl)
        except:
            continue  # skip if fit fails

    boot_vals = np.array(boot_vals)  # shape (samples, periods)
    lower = np.percentile(boot_vals, 2.5, axis=0)
    upper = np.percentile(boot_vals, 97.5, axis=0)

    for i, rp in enumerate(return_periods):
        z[rp]['low'] = lower[i]
        z[rp]['high'] = upper[i]

    return z


def pdfskill(obs, mod, bandwidth='scott', num_points=1000):
    """
    Computes the Perkins skill score — the area of overlap between the PDFs of obs and mod.
    
    Parameters:
        obs (array-like): Observed data (1D array).
        mod (array-like): Modeled data (1D array).
        bandwidth (str or float): Bandwidth for KDE. Options: 'scott', 'silverman', or a float.
        num_points (int): Number of evaluation points for the PDFs.
        
    Returns:
        float: Overlap skill score between 0 and 1.
    """
    obs = np.asarray(obs)
    mod = np.asarray(mod)

    obs = obs[~np.isnan(obs)]
    mod = mod[~np.isnan(mod)]

    combined = np.concatenate([obs, mod])
    x_eval = np.linspace(np.min(combined), np.max(combined), num_points)

    kde_obs = gaussian_kde(obs, bw_method=bandwidth)
    kde_mod = gaussian_kde(mod, bw_method=bandwidth)

    pdf_obs = kde_obs(x_eval)
    pdf_mod = kde_mod(x_eval)

    # Area of overlap
    overlap = np.trapz(np.minimum(pdf_obs, pdf_mod), x_eval)

    return overlap

def all_pdfskill(obs, mod, bandwidth='scott', num_points=1000):
    scores = []
    for i in range(obs.shape[1]):  # Loop over coordinates
        score = pdfskill(obs[:, i], mod[:, i], bandwidth=bandwidth, num_points=num_points)
        scores.append(score)
    return scores




def tailskill(obs, mod, threshold=0.95, bandwidth='scott', num_points=1000):
    """
    Computes a tail-focused skill score based on the area difference
    between the tails of two PDFs, weighted linearly outward.
    
    Parameters:
        obs (array-like): Observed data.
        mod (array-like): Modeled data.
        threshold (float): Probability threshold (e.g., 0.95 for upper tail).
        bandwidth (str or float): KDE bandwidth.
        num_points (int): Number of points for KDE evaluation.
    
    Returns:
        float: Tail skill score (0 = no skill, 1 = perfect match).
    """
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    obs = obs[~np.isnan(obs)]
    mod = mod[~np.isnan(mod)]

    if not (0 < threshold < 1):
        raise ValueError("Threshold must be between 0 and 1.")

    # Determine whether to use upper or lower tail
    lower_tail = threshold < 0.5

    # Common evaluation grid
    combined = np.concatenate([obs, mod])
    x_eval = np.linspace(np.min(combined), np.max(combined), num_points)

    kde_obs = gaussian_kde(obs, bw_method=bandwidth)
    kde_mod = gaussian_kde(mod, bw_method=bandwidth)

    pdf_obs = kde_obs(x_eval)
    pdf_mod = kde_mod(x_eval)

    # Estimate CDF of observed data from the PDF
    dx = x_eval[1] - x_eval[0]
    cdf_obs = np.cumsum(pdf_obs) * dx

    # Find the index at which the threshold CDF is exceeded
    if lower_tail:
        tail_idx = np.where(cdf_obs <= threshold)[0]
    else:
        tail_idx = np.where(cdf_obs >= threshold)[0]

    if len(tail_idx) == 0:
        return 0.0  # No tail region found

    # Slice the tail
    tail_slice = tail_idx if lower_tail else tail_idx

    tail_pdf_obs = pdf_obs[tail_slice]
    tail_pdf_mod = pdf_mod[tail_slice]
    tail_x = x_eval[tail_slice]

    # Compute absolute PDF difference
    diff = np.abs(tail_pdf_obs - tail_pdf_mod)

    # Weighting: linearly increasing outward
    weight = np.linspace(0, 1, len(diff)) * 10
    if lower_tail:
        weight = weight[::-1]

    weighted_diff = diff * weight
    total_weighted_error = np.sum(weighted_diff) * dx

    # Convert to skill score
    skill = 1 / (1 + total_weighted_error)

    return skill

def mean_tailskill(obs, mod, threshold=0.95):
    # Computes average tail skill across coordinates (columns)
    scores = []
    for i in range(obs.shape[1]):
        score = tailskill(mod[:, i], obs[:, i], threshold=threshold)
        scores.append(score)
    return np.nanmean(scores)