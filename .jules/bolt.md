
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.
## 2024-05-18 - VLM PIL Image Disk I/O Bottleneck
**Learning:** In Vision-Language Model pipelines (like using `AutoProcessor`), saving PIL Images to disk (`image.save("temp.jpg")`) just to pass a file path to the processor introduces significant synchronous disk I/O latency. Modern `transformers` processors natively support PIL Images directly in the payload.
**Action:** Always pass `PIL.Image` objects directly into the message payload (e.g., `{"type": "image", "image": pil_image}`) instead of writing temporary files to disk.
