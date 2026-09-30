
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2024-05-18 - Passing PIL Image directly to AutoProcessor
**Learning:** Writing frames to disk on every frame to pass to transformers' `AutoProcessor` causes massive synchronous disk I/O bottlenecks.
**Action:** Always pass `PIL.Image` objects directly in the message payload for Hugging Face VLM processors, avoiding unnecessary disk writes.
