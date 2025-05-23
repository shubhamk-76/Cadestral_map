import os
from Extracttext import TextExtractor, MapImageLoader
from visualize import visualize_ocr

if __name__ == '__main__':
    image_dir = '/home/shubham/EasyOCR_Project/Maps'
    output_csv = '/home/shubham/EasyOCR_Project/output.csv'
    output_vis_dir = '/home/shubham/EasyOCR_Project/ExtractedImage'
    loader = MapImageLoader(image_dir)
    image_paths = loader.get_image_paths()
    extractor = TextExtractor(use_gpu=False)
    extractor.process_all_images(image_paths, output_csv)
    for img_path in image_paths:
        save_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_vis_dir, f"{save_name}_overlay.jpg")
        visualize_ocr(img_path, save_path, extractor)
