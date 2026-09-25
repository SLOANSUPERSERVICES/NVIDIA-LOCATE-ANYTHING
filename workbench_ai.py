import cv2
import time
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

class WorkbenchAI:
    def __init__(self, model_path="nvidia/Eagle-2.5-8B", device="cpu", init_model=True):
        self.device = device
        self.init_model = init_model

        if init_model:
            print(f"Loading {model_path} on {device}...")
            # Use float32 on CPU or bfloat16 on GPU
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            self.processor.tokenizer.padding_side = "left"
            self.model = self.model.to(self.device)
            print("Model loaded.")
        else:
            print(f"Skipping model initialization for syntax testing (TEST_LOAD=0).")
            self.model = None
            self.processor = None

    def analyze_frame(self, image: Image.Image, prompt: str):
        if not self.init_model:
            return f"[Dummy Mode] LLM Report for prompt '{prompt}': The user is currently setting up the workbench according to the manual."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # transformers supports passing PIL image directly? Wait, inference.md passed URL/path.
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # In Eagle 2.5, image parameter might need to be path or URL or PIL Image?
        # Typically AutoProcessor can take PIL Image. Wait, inference doc:
        # It used: "image": "https://..." - but we can pass PIL Image if it supports it, or save to disk.
        # Let's save to a temp file to be safe, as it's definitely supported if we modify the messages to point to it, wait, transformers processor process_vision_info might expect URL/path.
        temp_image_path = "temp_frame.jpg"
        image.save(temp_image_path)
        messages[0]["content"][0]["image"] = temp_image_path

        text_list = [self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )]
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        inputs = self.processor(text=text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


def process_stream(feed_source, prompt, interval=5, init_model=False):
    ai = WorkbenchAI(init_model=init_model, device="cpu")

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

            print("\n--- Analyzing Current Frame ---")
            report = ai.analyze_frame(pil_image, prompt)
            print(f"Workbench AI Report:\n{report}\n")
            print("-" * 30)

            # Wait for next interval
            time.sleep(interval)

            # Flush buffer to get the latest frame after sleep (for live streams)
            # This is naive; in a real app, you might want to run capture in a separate thread
            # or grab multiple frames and discard them.
            if isinstance(feed_source, int) or str(feed_source).startswith("rtsp"):
                # Grab a few frames to clear the buffer
                for _ in range(5):
                    cap.grab()

    except KeyboardInterrupt:
        print("\nStopping workbench AI.")
    finally:
        cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Workbench AI - Live Stream LLM Analyzer")
    parser.add_argument("--feed", default=0, help="Video feed source: '0' for USB webcam, or an RTSP URL.")
    parser.add_argument("--prompt", default="Describe what you see on this workbench. Am I setting up the components correctly according to standard manuals?", help="Prompt for the LLM.")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds between analyses.")
    parser.add_argument("--test", action="store_true", help="Run in dummy mode without loading the model.")

    args = parser.parse_args()

    # Check if feed is integer (USB cam index)
    feed = int(args.feed) if str(args.feed).isdigit() else args.feed

    init_model = not args.test and os.environ.get("TEST_LOAD", "0") != "1"

    process_stream(feed, args.prompt, interval=args.interval, init_model=init_model)
