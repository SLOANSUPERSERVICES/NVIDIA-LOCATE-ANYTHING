
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.
## 2023-10-27 - Unnecessary Disk I/O Bottleneck
**Learning:** In live video pipeline scripts like `workbench_ai.py`, saving intermediate frames to disk (even temporary files like `temp_frame.jpg`) before passing them to the VLM processor introduces significant synchronous disk I/O overhead.
**Action:** `transformers` `AutoProcessor` natively accepts `PIL.Image` objects. Always pass image objects directly rather than writing them to disk first to maintain high throughput in stream processing.
