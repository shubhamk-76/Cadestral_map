import os
import cv2
import matplotlib.pyplot as plt
from Extracttext import TextExtractor

def visualize_ocr(img_path: str, save_path: str, extractor: TextExtractor, min_confidence: float = 0.5):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = extractor.reader.readtext(img_path)
    for (bbox, text, prob) in result:
        if prob >= min_confidence:
            (top_left, _, bottom_right, _) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
