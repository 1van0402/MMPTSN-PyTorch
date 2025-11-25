import torch
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo import osr
from scipy.interpolate import griddata
import os
import rasterio
from model import Model


def load_full_image_time_series_data(file_path):
    """加载全图时间序列数据"""
    df = pd.read_csv(file_path)
    geo_info = df.iloc[:, -2:].values
    data = df.iloc[:, :-7].values.astype("float32")

    time_steps = 12
    spectral_bands = 12
    spatial_bands = 3
    sar_bands = 2
    num_samples = data.shape[0]
    phe_data = df.iloc[:, -7:-2].values.astype("float32")
    spectral_data = []
    spatial_data = []
    sar_data = []

    for sample in range(num_samples):
        sample_data = data[sample, :]

        sample_spectral = []
        sample_spatial = []
        sample_sar = []

        for month in range(time_steps):
            start_idx = (month * (spectral_bands + spatial_bands + sar_bands))
            end_idx_spectral = start_idx + spectral_bands
            end_idx_sar =end_idx_spectral + sar_bands
            end_idx_spatial = end_idx_sar + spatial_bands

            sample_spectral.append(sample_data[start_idx:end_idx_spectral])
            sample_sar.append(sample_data[end_idx_spectral:end_idx_sar])
            sample_spatial.append(sample_data[end_idx_sar:end_idx_spatial])

        spectral_data.append(np.array(sample_spectral))
        spatial_data.append(np.array(sample_spatial))
        sar_data.append(np.array(sample_sar))

    spectral_data = np.array(spectral_data)
    spatial_data = np.array(spatial_data)
    sar_data = np.array(sar_data)

    print("spectral_data:", spectral_data.shape)
    print("spatial_data:", spatial_data.shape)
    print("sar_data:", sar_data.shape)
    print("phe_data:", phe_data.shape)

    return geo_info, spectral_data, spatial_data, sar_data, phe_data

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=(0, 1))
    if std is None:
        std = np.std(X, axis=(0, 1))
    return (X - mean) / (std + 1e-8), mean, std

def classify_samples_in_batches(model, spectral_data, spatial_data, sar_data, phe_data, batch_size=1024):
    model.eval()
    num_samples = spectral_data.shape[0]
    predictions = []

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_spectral_data = torch.from_numpy(spectral_data[start:end]).float().cuda()
            batch_spatial_data = torch.from_numpy(spatial_data[start:end]).float().cuda()
            batch_sar_data = torch.from_numpy(sar_data[start:end]).float().cuda()
            batch_phe_data = torch.from_numpy(phe_data[start:end]).float().cuda()

            outputs = model(batch_spectral_data, batch_spatial_data, batch_sar_data, batch_phe_data)
            _, predicted_classes = torch.max(outputs, 1)
            predictions.append(predicted_classes.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    unique_classes = np.unique(predictions)
    return predictions

def save_classification_as_geotiff(geo_info, classification_result, output_path, resolution=0.0001):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    min_lon, max_lon = geo_info[:, 0].min(), geo_info[:, 0].max()
    min_lat, max_lat = geo_info[:, 1].min(), geo_info[:, 1].max()

    x_grid = np.linspace(min_lon, max_lon, width)
    y_grid = np.linspace(min_lat, max_lat, height)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid[::-1])

    classification_grid = griddata(geo_info, classification_result, (x_mesh, y_mesh), method='linear')
    classification_grid = np.nan_to_num(classification_grid, nan=-1)

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Int16)

    if dataset is None:
        raise RuntimeError(f"Failed to create TIFF file at {output_path}. Check file path and permissions.")

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    dataset.SetProjection(srs.ExportToWkt())

    transform = [min_lon, (max_lon - min_lon) / width, 0, max_lat, 0, -(max_lat - min_lat) / height]
    dataset.SetGeoTransform(transform)

    dataset.GetRasterBand(1).WriteArray(classification_grid)

    dataset = None
    print(f"Classification results saved to {output_path}")

if __name__ == "__main__":
    file_path = './GuanzhaiSea/CSV/Predict-C.csv'
    geo_info, spectral_data, spatial_data, sar_data, phe_data = load_full_image_time_series_data(file_path)

    mean_spec, std_spec = None, None
    mean_spatial, std_spatial = None, None

    model = Model(
        spectral_dim=12,
        spatial_dim=3,
        sar_dim=2,
        phenology_dim=5,
        hidden_dims=256,
        num_layers=2,
        bidirectional=True,
        num_classes=3,
        dropout=0.5
    ).cuda()

    model.load_state_dict(torch.load('./GuanzhaiSea/Modelsave/globalC.pth', weights_only=True))

    batch_size = 1024
    classification_result = classify_samples_in_batches(model, spectral_data, spatial_data, sar_data, phe_data, batch_size)

    output_path = './GuanzhaiSea/Classification_results/C94.21.tif'
    width = 333
    height = 575
    # width = 884
    # height = 457

    # width = 499
    # height = 564
    with rasterio.open(output_path) as src:
        classification_grid = src.read(1)
        unique_values = np.unique(classification_grid)
        print(f"Unique values in saved TIFF: {unique_values}")
    save_classification_as_geotiff(geo_info, classification_result, output_path, resolution=0.0001)