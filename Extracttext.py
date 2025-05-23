import os
import re
import pandas as pd
import easyocr
from typing import List
from tqdm import tqdm


class MapImageLoader:
    def __init__(self, image_dir: str, extensions: List[str] = ['.jpg', '.png', '.jpeg']):
        self.image_dir = image_dir
        self.extensions = extensions

    def get_image_paths(self) -> List[str]:
        return [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in self.extensions
        ]

class TextExtractor:
    def __init__(self,use_gpu:bool = False): # we can make this true if we want to run on gpu.
        self.reader = easyocr.Reader(['en'])
        self.number_pattern = re.compile(r'^\d+$')

    def recognize_text(self, img_path: str) -> List[dict]:
        results = self.reader.readtext(img_path)
        rows = []
        for _, text, _ in results:
            text = text.strip().replace(',', '').replace('\n', ' ')
            if not text:
                continue
            if self.number_pattern.fullmatch(text):  # Only numbers
                rows.append({'Character': '', 'Number': text})
            else:
                rows.append({'Character': text, 'Number': ''})
        return rows

    def process_all_images(self, image_paths: List[str], output_csv: str = 'output.csv'):
        all_rows = []
        for img_path in tqdm(image_paths, desc='Processing images'):
            rows = self.recognize_text(img_path)
            all_rows.extend(rows)
        df = pd.DataFrame(all_rows, columns=['Character', 'Number'])
        df.to_csv(output_csv, index=False)
        print(f"All text extracted and saved to {output_csv}")
