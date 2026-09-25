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

        # Performance optimization: Only convert to RGB if not already in that mode
        # to prevent unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")

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

        # Performance optimization: Only convert to RGB if not already in that mode
        # to prevent unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")

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

        # Performance optimization: Only convert to RGB if not already in that mode
        # to prevent unnecessary memory copying
        img = Image.open(image_path)
        img = img if img.mode == "RGB" else img.convert("RGB")

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

def main():
    print("--- LocateAnything Integration Example ---")

    # Check if the environment has enough memory/CUDA for real initialization
    # To prevent OOM kill during unit testing on constrained machines, we can skip full model load
    init_model = os.environ.get("TEST_LOAD", "0") == "1"

    app = VisionApp(device="cpu", init_model=init_model)

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

    finally:
        # Clean up dummy image
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)

if __name__ == "__main__":
    main()
