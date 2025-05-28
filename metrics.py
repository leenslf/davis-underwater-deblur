import cv2
import numpy as np
from skimage import img_as_float, filters, feature, segmentation
from skimage.measure import shannon_entropy
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
import os

# Update these with the paths to your images
image_paths = {
    "Raw RGB": "assets/raw.png",
    "Static Fusion": "assets/deblurred.png",
    "Temporal Fusion": "assets/firstexp.png"
}

def variance_of_laplacian(image):
    """Original sharpness metric - variance of Laplacian"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def entropy(image):
    """Original entropy metric - Shannon entropy"""
    return shannon_entropy(img_as_float(image))

def gradient_magnitude_variance(image):
    """Sharpness based on gradient magnitude variance"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.var(gradient_magnitude)

def tenengrad_variance(image):
    """Tenengrad focus measure - variance of Sobel gradient"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobelx**2 + sobely**2
    return np.var(tenengrad)

def brenner_gradient(image):
    """Brenner's focus measure"""
    dx = np.diff(image.astype(np.float64), axis=1)
    dy = np.diff(image.astype(np.float64), axis=0)
    # Pad to maintain original size
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    return np.mean(dx**2 + dy**2)

def edge_density(image):
    """Density of edge pixels using Canny edge detection"""
    edges = cv2.Canny(image, 50, 150)
    return np.sum(edges > 0) / edges.size

def local_contrast(image):
    """Local contrast using standard deviation in local neighborhoods"""
    kernel = np.ones((3, 3), np.float32) / 9
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((image.astype(np.float32) - local_mean)**2, -1, kernel)
    local_std = np.sqrt(local_variance)
    return np.mean(local_std)

def michelson_contrast(image):
    """Michelson contrast - (max - min) / (max + min)"""
    max_val = np.max(image)
    min_val = np.min(image)
    if max_val + min_val == 0:
        return 0
    return (max_val - min_val) / (max_val + min_val)

def rms_contrast(image):
    """Root Mean Square contrast"""
    mean_intensity = np.mean(image)
    return np.sqrt(np.mean((image - mean_intensity)**2))

def histogram_entropy(image):
    """Entropy based on histogram distribution"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Remove zero bins
    prob = hist / np.sum(hist)
    return scipy_entropy(prob, base=2)

def spatial_frequency(image):
    """Spatial frequency measure"""
    image_float = image.astype(np.float64)
    # Row frequency
    RF = np.sqrt(np.mean(np.diff(image_float, axis=1)**2))
    # Column frequency
    CF = np.sqrt(np.mean(np.diff(image_float, axis=0)**2))
    return np.sqrt(RF**2 + CF**2)

def average_gradient(image):
    """Average gradient magnitude"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(gradient_magnitude)

def image_power(image):
    """Image power in frequency domain"""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    return np.mean(magnitude_spectrum**2)

def high_frequency_content(image):
    """High frequency content measure"""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    h, w = image.shape
    # Create high-pass filter (remove low frequencies in center)
    center_h, center_w = h // 2, w // 2
    mask = np.ones((h, w))
    r = min(h, w) // 8  # Radius for low frequency removal
    y, x = np.ogrid[:h, :w]
    mask_area = (x - center_w)**2 + (y - center_h)**2 <= r**2
    mask[mask_area] = 0
    
    high_freq = f_shift * mask
    return np.mean(np.abs(high_freq)**2)

def structure_tensor_coherence(image):
    """Coherence measure based on structure tensor"""
    # Compute gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian smoothing
    sigma = 1.0
    Ixx = ndimage.gaussian_filter(Ixx, sigma)
    Iyy = ndimage.gaussian_filter(Iyy, sigma)
    Ixy = ndimage.gaussian_filter(Ixy, sigma)
    
    # Compute coherence
    coherence = ((Ixx - Iyy)**2 + 4*Ixy**2) / ((Ixx + Iyy)**2 + 1e-10)
    return np.mean(coherence)

results = []

for name, path in image_paths.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load: {path}")
        continue

    # Calculate all metrics
    metrics = {
        'Laplacian Variance': variance_of_laplacian(image),
        'Shannon Entropy': entropy(image),
        'Gradient Magnitude Var': gradient_magnitude_variance(image),
        'Tenengrad Variance': tenengrad_variance(image),
        'Brenner Gradient': brenner_gradient(image),
        'Edge Density': edge_density(image),
        'Local Contrast': local_contrast(image),
        'Michelson Contrast': michelson_contrast(image),
        'RMS Contrast': rms_contrast(image),
        'Histogram Entropy': histogram_entropy(image),
        'Spatial Frequency': spatial_frequency(image),
        'Average Gradient': average_gradient(image),
        'Image Power': image_power(image),
        'High Freq Content': high_frequency_content(image),
        'Structure Coherence': structure_tensor_coherence(image)
    }
    
    results.append((name, metrics))

# Print results in a formatted table
print("\nComprehensive No-Reference Image Quality Metrics:")
print("=" * 120)

# Print header
metric_names = list(results[0][1].keys())
print(f"{'Image':<20}", end="")
for metric in metric_names:
    print(f"{metric:<15}", end="")
print()
print("-" * 120)

# Print results
for name, metrics in results:
    print(f"{name:<20}", end="")
    for metric_name in metric_names:
        value = metrics[metric_name]
        if value > 1000:
            print(f"{value:<15.2e}", end="")
        else:
            print(f"{value:<15.4f}", end="")
    print()

# Print metric descriptions
print("\n" + "=" * 120)
print("METRIC DESCRIPTIONS:")
print("=" * 120)
descriptions = {
    'Laplacian Variance': 'Sharpness measure - higher values indicate sharper images',
    'Shannon Entropy': 'Information content - higher values indicate more complex images',
    'Gradient Magnitude Var': 'Variance of gradient magnitudes - measures edge strength variation',
    'Tenengrad Variance': 'Variance of squared gradients - focus quality measure',
    'Brenner Gradient': 'Average squared gradient - sharpness measure',
    'Edge Density': 'Proportion of edge pixels - measures edge richness',
    'Local Contrast': 'Average local standard deviation - texture measure',
    'Michelson Contrast': 'Global contrast measure (max-min)/(max+min)',
    'RMS Contrast': 'Root mean square contrast - overall contrast measure',
    'Histogram Entropy': 'Entropy of intensity histogram - tonal distribution',
    'Spatial Frequency': 'Overall spatial activity measure',
    'Average Gradient': 'Mean gradient magnitude - edge strength',
    'Image Power': 'Power in frequency domain - overall energy',
    'High Freq Content': 'High frequency energy - detail richness',
    'Structure Coherence': 'Local structure organization measure'
}

for metric, description in descriptions.items():
    print(f"{metric:<20}: {description}")

print("\n" + "=" * 120)
print("INTERPRETATION GUIDELINES:")
print("- Higher sharpness metrics (Laplacian, Gradient, Tenengrad, Brenner) = better focus")
print("- Higher entropy metrics = more information/complexity")
print("- Higher contrast metrics = better visual quality")
print("- Higher edge density = more detail preservation")
print("- Compare relative values between your processed images")
print("=" * 120)