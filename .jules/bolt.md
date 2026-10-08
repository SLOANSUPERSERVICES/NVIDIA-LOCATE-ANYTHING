
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2023-10-27 - Synchronous Disk I/O in VLM Pipelines
**Learning:** Writing frames to disk synchronously (e.g., `image.save("temp_frame.jpg")`) on a live video stream to pass to an LLM/VLM pipeline introduces massive, unnecessary latency. Hugging Face `AutoProcessor` can natively accept `PIL.Image` objects in its message schema.
**Action:** Always pass `PIL.Image` objects directly into the message content dictionary for local/HuggingFace processors instead of staging them on the file system.
