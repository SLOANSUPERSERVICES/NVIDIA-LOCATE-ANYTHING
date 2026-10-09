
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2024-05-18 - Disk I/O Bottleneck with Vision-Language Models
**Learning:** Saving a `PIL.Image` object to a temporary file in order to pass its path to a Hugging Face `AutoProcessor` creates a massive synchronous disk I/O bottleneck, heavily degrading frame processing rates in real-time video streaming pipelines.
**Action:** When initializing Vision-Language Models (e.g., Eagle 2.5), ensure `PIL.Image` objects are passed directly into the message schema's `image` key to eliminate disk writes and memory serialization overhead.
