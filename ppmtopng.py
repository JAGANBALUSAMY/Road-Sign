import os
from PIL import Image

# Input folder containing .ppm images
input_folder = "E:/FS Project/GTSRB/Final_Test/Images"  # Change this to your folder path
output_folder = "E:/FS Project/uploads"  # Change this to where you want PNGs saved

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert all .ppm files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".ppm"):  # Process only .ppm files
        ppm_path = os.path.join(input_folder, filename)
        png_path = os.path.join(output_folder, filename.replace(".ppm", ".png"))

        # Open and convert the image
        img = Image.open(ppm_path)
        img.save(png_path, "PNG")

        print(f"Converted: {ppm_path} -> {png_path}")

print("✅ Conversion completed!")
