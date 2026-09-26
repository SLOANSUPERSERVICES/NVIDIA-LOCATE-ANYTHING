import json
import argparse
from typing import List, Dict

def create_barcode_sample(image_path: str, barcodes: List[Dict]) -> Dict:
    """
    Format a barcode sample for LocateAnything training.
    barcodes format: [{"label": "123456", "box": [x1, y1, x2, y2]}, ...]
    (coordinates should be normalized 0-1000)
    """
    prompt = "Detect and decode all barcodes and QR codes in the image."
    answer = ""
    for b in barcodes:
        label = b["label"]
        x1, y1, x2, y2 = b["box"]
        answer += f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"

    if not answer:
        answer = "<box>none</box>"

    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer}
        ],
        "image": image_path
    }

def create_shipping_label_sample(image_path: str, labels: List[Dict]) -> Dict:
    """
    Format a shipping label sample for LocateAnything training.
    labels format: [{"label_type": "tracking_number", "content": "XYZ123", "box": [x1, y1, x2, y2]}, ...]
    (coordinates should be normalized 0-1000)
    """
    prompt = "Read and parse all shipping labels in the image, including parcels and pallets. Extract relevant information like barcodes, tracking numbers, and addresses."
    answer = ""
    for lbl in labels:
        content = f"{lbl['label_type']}: {lbl['content']}"
        x1, y1, x2, y2 = lbl["box"]
        answer += f"<ref>{content}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"

    if not answer:
        answer = "<box>none</box>"

    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer}
        ],
        "image": image_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample training data for LocateAnything Logistics Tasks")
    parser.add_argument("--output", type=str, default="data/annotations/sample_logistics.jsonl")
    args = parser.parse_args()

    # Dummy example generation
    sample_barcodes = [
        {"label": "123456789012", "box": [100, 200, 300, 250]}
    ]
    sample_labels = [
        {"label_type": "tracking_number", "content": "1Z9999999999999999", "box": [500, 100, 800, 150]},
        {"label_type": "address", "content": "123 Main St", "box": [500, 200, 700, 250]}
    ]

    res1 = create_barcode_sample("barcodes/test_img1.jpg", sample_barcodes)
    res2 = create_shipping_label_sample("parcels/test_img2.jpg", sample_labels)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(json.dumps(res1) + "\n")
        f.write(json.dumps(res2) + "\n")

    print(f"Sample data written to {args.output}")
