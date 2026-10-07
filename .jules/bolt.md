
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2024-05-18 - Passing PIL Images to HF AutoProcessor
**Learning:** Writing PIL Image frames to disk and passing the file path to `AutoProcessor` in a loop (like `workbench_ai.py`) introduces massive synchronous disk I/O bottlenecks.
**Action:** Always pass the `PIL.Image` object directly into the payload array (`messages[0]["content"][0]["image"] = image`) to skip disk I/O, as Hugging Face processors natively support this.
