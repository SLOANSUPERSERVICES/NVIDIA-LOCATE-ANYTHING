## 2024-05-18 - Model output parsing performance
**Learning:** Dense point and bounding box parsing from string outputs of the LocateAnything model scales poorly due to continuous regex recompilation and per-coordinate division math.
**Action:** When extracting values from string formats in Python, pre-compile regular expressions at the module scope and hoist expensive calculations out of the loop.

## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.
