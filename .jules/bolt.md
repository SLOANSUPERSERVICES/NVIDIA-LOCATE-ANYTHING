
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2024-11-20 - PIL Image Support in AutoProcessor
**Learning:** In VLM pipelines, writing intermediate frames to disk before passing them to the model causes a massive bottleneck. We discovered that `transformers` `AutoProcessor` can natively accept `PIL.Image` objects directly in the message payload.
**Action:** Always pass `PIL.Image` objects directly in VLM message arrays instead of writing temporary files, avoiding costly synchronous disk I/O.
