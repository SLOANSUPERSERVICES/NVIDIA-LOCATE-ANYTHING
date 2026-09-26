
## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.

## 2023-11-20 - Unconditional PIL Image Conversion (Follow-up)
**Learning:** Found that the performance bottleneck where `Image.convert("RGB")` performs redundant copying is quite prevalent throughout the codebase's data loading (e.g. `Embodied/eaglevl/train/tools.py`) and inference logic (e.g. `Embodied/evaluation/*`). Conditional checking completely bypasses the copying step overhead.
**Action:** Always scan for unconditional `convert` operations on PIL Images when aiming for quick performance gains during IO-heavy workflows.
