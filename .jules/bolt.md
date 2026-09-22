## 2024-05-20 - Re.finditer is slow, caching Regex compiles speeds up parsing
**Learning:** `re.finditer(r"...", answer)` is significantly slower than pre-compiling the regex in Python. Since `parse_boxes` and `parse_points` are called repeatedly for model output parsing, the `re.compile()` should be lifted to module level. Extracting integers from `m.groups()` is also slightly faster.
**Action:** Always pre-compile regex patterns at the module level when they are used in tight loops or parsing functions called repeatedly. Pre-compute constant math ratios like `w / 1000.0` outside of regex parsing loops.

## 2023-10-27 - PIL Image Conversion Overhead
**Learning:** Unconditionally calling `.convert("RGB")` on PIL images in a hot path can cause massive performance hits due to memory copying, even if the image is already in RGB mode (e.g., dropping execution time from ~900ms to near 0ms for 4K images).
**Action:** Always wrap `.convert("RGB")` with a mode check: `image if image.mode == "RGB" else image.convert("RGB")`.
