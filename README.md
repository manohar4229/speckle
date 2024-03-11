

**Title: Image Speckle Noise Analysis**

**Purpose:**

This Python code analyzes images within a directory structure to identify potential speckle noise. It provides informative statistics and optionally writes results to separate CSV files for noisy and clean images.

**Functionality:**

1. **Image Loading and Grayscaling:**
   - Loads grayscale images from specified folders (`train` and `valid`) with subfolders (`yes` and `no`) using OpenCV (`cv2`).
   - Handles potential loading errors gracefully.

2. **Speckle Noise Detection:**
   - Employs the `is_potential_speckle_noise` function to assess image characteristics suggestive of speckle noise using:
     - Median filtering (optional) for noise reduction (adjust parameters if needed).
     - Standard deviation and mean calculation to derive a noise ratio.
     - Otsu's thresholding (optional) for basic noise detection.
     - A combined score based on noise ratio and thresholding for more robust results.
   - Returns a boolean indicating potential speckle noise (`is_noisy`), along with additional statistics:
     - Noise ratio
     - Coefficient of variation
     - Minimum intensity
     - Maximum intensity

3. **Optional Speckle Noise Addition:**
   - If `add_noise_to_clean` is set to `True`, images classified as not having noise undergo speckle noise addition using the `add_speckle_noise` function, which controls noise intensity via the `noise_amount` parameter.
   - The noisy versions are saved with a "noisy_" prefix in the original image directory.

4. **Statistical Analysis and CSV Output:**
   - Calculates and prints informative statistics:
     - Total images analyzed
     - Images with potential speckle noise (percentage)
     - Images without potential speckle noise (percentage)
     - Average noise ratio (for noisy images only)
   - Optionally writes noisy image data (filename, is_noisy, noise ratio, and other statistics) to separate CSV files for each split (`train` and `valid`) and label (`yes` and `no`) using the `write_results_to_csv` function.
   - The CSV filenames follow the format `{split}_{label}_noisy.csv` (e.g., `train_yes_noisy.csv`).
   - You can uncomment the code block to write clean image data to separate CSV files following the format `{split}_{label}_clean.csv`.

**Usage:**

```python
import read_images_and_analyze_noise

# Replace with your data directory path
data_dir = "path/to/your/data"

# Optional: Dictionary mapping labels to class names (default: {'yes': 0, 'no': 1})
label_to_class_name = {'yes': 'Brain Tumor', 'no': 'Normal'}

# Optional: Add speckle noise to clean images (default: False)
add_noise_to_clean = False

# Optional: Control noise intensity when adding speckle noise (default: 0.1)
noise_amount = 0.1

read_images_and_analyze_noise(data_dir, label_to_class_name, add_noise_to_clean, noise_amount)
```

**Dependencies:**

- NumPy (`numpy`)
- OpenCV (`cv2`)
- SciPy (`scipy.ndimage` for median filtering)
- Scikit-image (`skimage.filters` for Otsu's thresholding)

**Customization:**

- Adjust noise detection parameters in `is_potential_speckle_noise` as needed for your specific dataset.
- Modify the `output_dir` parameter in `read_images_and_analyze_noise` to specify a custom directory for CSV files.

**Additional Notes:**

- Consider experimenting with different noise detection techniques for optimal results on your data.
- Explore noise visualization methods for further analysis.

I hope this enhanced README provides a clear and informative overview of the code!
