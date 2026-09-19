## 2024-05-18 - Model output parsing performance
**Learning:** Dense point and bounding box parsing from string outputs of the LocateAnything model scales poorly due to continuous regex recompilation and per-coordinate division math.
**Action:** When extracting values from string formats in Python, pre-compile regular expressions at the module scope and hoist expensive calculations out of the loop.
