import csv
import numpy as np
import os
from scipy.ndimage import median_filter  # For noise reduction
from skimage.filters import threshold_otsu  # For basic speckle noise detection
import cv2  # For image loading


def is_potential_speckle_noise(image, threshold=0.1, otsu_weight=0.5):
    image = median_filter(image, size=(3, 3)) 
    std = np.std(image.flatten())
    mean = np.mean(image.flatten())
    noise_ratio = std / mean
    otsu_threshold = threshold_otsu(image)
    binary_image = image > otsu_threshold
    combined_score = (otsu_weight * noise_ratio) + ((1 - otsu_weight) * np.mean(binary_image))
    is_noisy = combined_score > threshold

    min_intensity = np.min(image)
    max_intensity = np.max(image)
    coefficient_of_variation = std / mean if mean > 0 else 0  # Avoid division by zero

    return is_noisy, noise_ratio, coefficient_of_variation, min_intensity, max_intensity


def load_image_grayscale(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Failed to load image '{image_path}'.")
            return None
        return image
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None


def add_speckle_noise(image, amount=0.1):
    noise = np.random.rand(*image.shape) * amount + 1
    noisy_image = image * noise

    return noisy_image.astype(np.uint8) 

def write_results_to_csv(results, filename, header_row=None):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_row) if header_row else csv.writer(csvfile)
        if header_row:
            writer.writeheader()
        writer.writerows(results)


def read_images_and_analyze_noise(data_dir, label_to_class_name={'yes': 0, 'no': 1},
                                 add_noise_to_clean=False, noise_amount=0.1,
                                 output_dir="analysis_results"):

    os.makedirs(output_dir, exist_ok=True) 

    for split in ['train', 'valid']:  
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: Split directory '{split_dir}' not found.")
            continue

        for label, class_name in label_to_class_name.items():
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                print(f"Warning: Label directory '{label_dir}' not found.")
                continue

            noisy_results = []  
            clean_results = []
            total_images = 0
            noisy_images = 0
            clean_images = 0

            for filename in os.listdir(label_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(label_dir, filename)
                image = load_image_grayscale(image_path)
                if image is None:
                    continue

                is_noisy, noise_ratio, coefficient_of_variation, min_intensity, max_intensity = \
                    is_potential_speckle_noise(image)

                total_images += 1
                if is_noisy:
                    noisy_images += 1
                    noisy_results.append({
                        "filename": filename,
                        "is_noisy": is_noisy,
                        "noise_ratio": noise_ratio,
                        "coefficient_of_variation": coefficient_of_variation,
                        "min_intensity": min_intensity,
                        "max_intensity": max_intensity
                    })
                else:
                    clean_images += 1
                    clean_results.append({
                        "filename": filename,
                        "is_noisy": is_noisy,
                        "noise_ratio": noise_ratio,
                        "coefficient_of_variation": coefficient_of_variation,
                        "min_intensity": min_intensity,
                        "max_intensity": max_intensity
                    })


                    if add_noise_to_clean:
                        noisy_image = add_speckle_noise(image.copy(), noise_amount)                       
                        cv2.imwrite(os.path.join(label_dir, f"noisy_{filename}"), noisy_image)
                        print(f"  - Added speckle noise to clean image: {filename}")

    
            if total_images > 0:
                noise_percentage = (noisy_images / total_images) * 100
                clean_percentage = (clean_images / total_images) * 100
                average_noise_ratio = np.mean([result["noise_ratio"] for result in noisy_results]) if noisy_images > 0 else 0
                average_coefficient_of_variation = np.mean([result["coefficient_of_variation"] for result in noisy_results]) if noisy_images > 0 else 0

                print(f"Split: {split} - Label: {label} ({class_name})")
                print(f"- Images Analyzed: {total_images}")
                print(f"- Images with Potential Speckle Noise: {noisy_images} ({noise_percentage:.2f}%)")
                print(f"- Images without Potential Speckle Noise: {clean_images} ({clean_percentage:.2f}%)")
                if noisy_images > 0:
                    print(f"- Average Noise Ratio: {average_noise_ratio:.4f}")

                
                if noisy_results:
                    header_row = ["filename", "is_noisy", "noise_ratio", "coefficient_of_variation", "min_intensity", "max_intensity"]
                    noisy_csv_path = os.path.join(output_dir, f"{split}_{label}_noisy.csv")
                    write_results_to_csv(noisy_results, noisy_csv_path, header_row)

                
                if clean_results:
                    header_row = ["filename", "is_noisy", "noise_ratio", "coefficient_of_variation", "min_intensity", "max_intensity"]
                    clean_csv_path = os.path.join(output_dir, f"{split}_{label}_clean.csv")
                    write_results_to_csv(clean_results, clean_csv_path, header_row)

# data_dir = "dataset"  
# label_to_class_name = {'yes': 0, 'no': 1}
# read_images_and_analyze_noise(data_dir, label_to_class_name, add_noise_to_clean=True, noise_amount=0.1)