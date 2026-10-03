
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2026-10-03 - VLM Pipeline Synchronous Disk I/O Bottleneck
**Learning:** Writing temporary image frames to disk (e.g., using `image.save('temp.jpg')`) before passing them to a Hugging Face `AutoProcessor` creates a major synchronous disk I/O bottleneck in VLM pipelines.
**Action:** Pass `PIL.Image` objects directly in the message payload instead of saving temporary frames to disk, as `AutoProcessor` natively supports this. This eliminates the I/O overhead.
