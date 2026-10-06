
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2024-11-20 - VLM Inference Disk I/O Overhead
**Learning:** Saving temporary images to disk before passing them to a Vision-Language Model's `AutoProcessor` creates significant synchronous disk I/O overhead.
**Action:** Always pass `PIL.Image` objects directly into the message payload for Hugging Face `AutoProcessor` when working with Vision-Language Models to avoid the disk write bottleneck.
