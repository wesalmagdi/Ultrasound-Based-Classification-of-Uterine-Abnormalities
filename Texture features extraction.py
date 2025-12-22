import os
import numpy as np
import pandas as pd
import tempfile
import imageio.v3 as iio
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from radiomics import featureextractor

IMG_DIR = "data/images"
MASK_DIR = "data/predicted_masks"

extractor = featureextractor.RadiomicsFeatureExtractor("radiomics_params.yaml")

texture_list = []
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

# Local Binary Patterns (LBP): Compares a pixel's intensity to its neighbors to create a binary pattern.
# Wavelet Transforms: Decompose images into different frequency components to capture multi-scale texture
image_files = sorted(os.listdir(IMG_DIR))
for filename in image_files:
    patient_id = os.path.splitext(filename)[0]
    img_path = os.path.join(IMG_DIR, filename)

    mask_candidates = [f"{patient_id}_mask.png", f"{patient_id}_pred_mask.png"]
    mask_path = None
    for m in mask_candidates:
        p = os.path.join(MASK_DIR, m)
        if os.path.exists(p):
            mask_path = p
            break
    if mask_path is None:
        print(f"No mask found for {patient_id}, skipping.")
        continue

    mask_bin = imread(mask_path)
    if mask_bin.ndim == 3:
        mask_bin = mask_bin[:, :, 0]
    mask_bin = np.where(mask_bin > 0, 1, 0)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
        tmp_mask_path = tmp_mask.name
    iio.imwrite(tmp_mask_path, mask_bin.astype(np.uint8))

    features_raw = extractor.execute(img_path, tmp_mask_path, label=1)
    os.remove(tmp_mask_path)

    texture_features = {k: v for k, v in features_raw.items() 
                        if any(t in k for t in ["glcm", "glrlm", "glszm", "gldm", "ngtdm"])}

    img = imread(img_path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img_as_ubyte(img)

    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    lbp_region = lbp[mask_bin.astype(bool)]
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp_region, bins=n_bins, range=(0, n_bins), density=True)

    lbp_features = {f"LBP_hist_bin_{i}": hist[i] for i in range(len(hist))}
    lbp_features["LBP_mean"] = lbp_region.mean()
    lbp_features["LBP_std"] = lbp_region.std()
    lbp_features["LBP_entropy"] = -np.sum(hist * np.log2(hist + 1e-10))

    texture_list.append({"id": patient_id, **texture_features, **lbp_features})

df_texture = pd.DataFrame(texture_list)
df_texture["id"] = df_texture["id"].astype(int)
LABELS_CSV = "data/labels.csv"

if os.path.exists(LABELS_CSV):
    df_labels = pd.read_csv(LABELS_CSV)
    df_combined = pd.merge(df_labels, df_texture, on="id", how="left")
else:
    df_combined = df_texture

df_combined.to_csv(LABELS_CSV, index=False)
print(f"Saved/Updated: {LABELS_CSV}")

