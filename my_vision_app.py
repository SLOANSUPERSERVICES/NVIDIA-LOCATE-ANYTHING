import sys
import os

# Ensure the Embodied directory is in the Python path
# This allows importing the LocateAnything worker from the repo structure
sys.path.append(os.path.join(os.path.dirname(__file__), "Embodied"))

import torch
from PIL import Image

try:
    from locateanything_worker import LocateAnythingWorker
except ImportError as e:
    print(f"Failed to import LocateAnythingWorker. Ensure you are in the correct directory. Error: {e}")
    sys.exit(1)

class VisionApp:
    """
    A template class demonstrating how to integrate LocateAnything into your vision applications.
    """
    def __init__(self, model_path="nvidia/LocateAnything-3B", device="cpu", init_model=True):
        print(f"Initializing LocateAnything Worker with model: {model_path} on {device}...")
        # Note: we use float32 here if on CPU to avoid bfloat16 errors on non-Ampere CPUs.
        # If you have a GPU, use device="cuda" and dtype=torch.bfloat16.
        if init_model:
            self.worker = LocateAnythingWorker(
                model_path,
                device=device,
                dtype=torch.float32 if device == "cpu" else torch.bfloat16
            )
            print("Initialization complete.")
        else:
            self.worker = None
            print("Initialization skipped for syntax testing.")

    def detect_objects(self, image_path, categories):
        """
        Detect bounding boxes for specified categories.
        """
        if self.worker is None:
            return [{"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}] # Dummy result

        # ⚡ Bolt: conditional RGB conversion to avoid unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")
        print(f"Running detection for categories: {categories}")
        result = self.worker.detect(img, categories)

        w, h = img.size
        boxes = self.worker.parse_boxes(result["answer"], w, h)
        return boxes

    def point_at_object(self, image_path, phrase):
        """
        Get a point coordinate for a specific object description.
        """
        if self.worker is None:
            return [{"x": 50.0, "y": 50.0}] # Dummy result

        # ⚡ Bolt: conditional RGB conversion to avoid unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")
        print(f"Pointing at: '{phrase}'")
        result = self.worker.point(img, phrase)

        w, h = img.size
        points = self.worker.parse_points(result["answer"], w, h)
        return points

    def ground_gui_element(self, image_path, phrase, output_type="box"):
        """
        Ground a GUI element (like a button or icon) by returning a box or point.
        """
        if self.worker is None:
            return [{"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}] if output_type == "box" else [{"x": 50.0, "y": 50.0}]

        # ⚡ Bolt: conditional RGB conversion to avoid unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")
        print(f"Grounding GUI element: '{phrase}' as {output_type}")
        result = self.worker.ground_gui(img, phrase, output_type=output_type)

        w, h = img.size
        if output_type == "box":
            return self.worker.parse_boxes(result["answer"], w, h)
        elif output_type == "point":
            return self.worker.parse_points(result["answer"], w, h)

    def decode_barcodes(self, image_path):
        """
        Detect and decode all barcodes and QR codes in the image.
        """
        if self.worker is None:
            return [{"barcode_data": "dummy123", "box": {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}}] # Dummy result

        # ⚡ Bolt: conditional RGB conversion to avoid unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")
        print(f"Decoding barcodes in {image_path}")
        result = self.worker.decode_barcodes(img)
        return result["answer"]

    def read_shipping_labels(self, image_path):
        """
        Read and parse all shipping labels in the image, including parcels and pallets.
        """
        if self.worker is None:
            return [{"tracking_number": "TRK987654321", "box": {"x1": 10.0, "y1": 10.0, "x2": 150.0, "y2": 150.0}}] # Dummy result

        # ⚡ Bolt: conditional RGB conversion to avoid unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")
        print(f"Reading shipping labels in {image_path}")
        result = self.worker.read_shipping_labels(img)
        return result["answer"]

import argparse
import time
import cv2

def process_stream(app: VisionApp, feed_source, interval=5):
    print(f"Opening feed: {feed_source}")
    cap = cv2.VideoCapture(feed_source)

    if not cap.isOpened():
        print(f"Failed to open video feed: {feed_source}")
        return

    print(f"Starting analysis every {interval} seconds. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot read frame.")
                break

            # Convert OpenCV frame (BGR) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Save temporarily for app methods if needed, or we can update app methods to take PIL Image.
            # But the existing app methods take image_path.
            temp_image_path = "temp_stream_frame.jpg"
            pil_image.save(temp_image_path)

            print("\n--- Analyzing Current Frame ---")

            barcodes = app.decode_barcodes(temp_image_path)
            print(f"Barcode Results: {barcodes}")

            shipping_labels = app.read_shipping_labels(temp_image_path)
            print(f"Shipping Label Results: {shipping_labels}")

            print("-" * 30)

            # Wait for next interval
            time.sleep(interval)

            # Flush buffer to get the latest frame after sleep (for live streams)
            if isinstance(feed_source, int) or str(feed_source).startswith("rtsp"):
                for _ in range(5):
                    cap.grab()

    except KeyboardInterrupt:
        print("\nStopping vision app.")
    finally:
        cap.release()
        if os.path.exists("temp_stream_frame.jpg"):
            os.remove("temp_stream_frame.jpg")

def main():
    parser = argparse.ArgumentParser(description="Vision App - Live Stream & Dummy Tests")
    parser.add_argument("--feed", default=None, help="Video feed source: '0' for USB webcam, or an RTSP URL. If not provided, runs dummy tests.")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between analyses for live feed.")
    parser.add_argument("--test", action="store_true", help="Run in dummy mode without loading the model.")

    args = parser.parse_args()

    print("--- LocateAnything Integration Example ---")

    # Check if the environment has enough memory/CUDA for real initialization
    # To prevent OOM kill during unit testing on constrained machines, we can skip full model load
    init_model = not args.test and os.environ.get("TEST_LOAD", "0") == "1"

    app = VisionApp(device="cpu", init_model=init_model)

    if args.feed is not None:
        feed = int(args.feed) if str(args.feed).isdigit() else args.feed
        process_stream(app, feed, interval=args.interval)
    else:
        # Create a dummy image for testing
        dummy_image_path = "dummy_test_image.jpg"
        img = Image.new('RGB', (800, 600), color = 'white')
        img.save(dummy_image_path)

        try:
            # Example 1: Detection
            boxes = app.detect_objects(dummy_image_path, ["person", "car"])
            print(f"Detection Results: {boxes}")

            # Example 2: Pointing
            points = app.point_at_object(dummy_image_path, "the red car")
            print(f"Pointing Results: {points}")

            # Example 3: GUI Grounding
            gui_element = app.ground_gui_element(dummy_image_path, "the search button", output_type="box")
            print(f"GUI Grounding Results: {gui_element}")

            # Example 4: Decode Barcodes
            barcodes = app.decode_barcodes(dummy_image_path)
            print(f"Barcode Results: {barcodes}")

            # Example 5: Read Shipping Labels
            shipping_labels = app.read_shipping_labels(dummy_image_path)
            print(f"Shipping Label Results: {shipping_labels}")

        finally:
            # Clean up dummy image
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)

if __name__ == "__main__":
    main()
