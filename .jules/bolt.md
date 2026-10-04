
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2025-05-26 - Synchronous Disk I/O in VLM Pipelines
**Learning:** In Vision-Language Model pipelines (like `workbench_ai.py` using Hugging Face `AutoProcessor`), writing temporary frames to disk creates a severe synchronous I/O bottleneck, especially for live video streams. The `transformers` `AutoProcessor` natively supports `PIL.Image` objects in its chat template format.
**Action:** When working with image streams and `AutoProcessor`, pass `PIL.Image` objects directly into the `messages` array instead of saving to disk to avoid disk write overhead.
