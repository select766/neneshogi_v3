"""
Register neneshogi C++ python module to import path
Add __init__.py to module directory
"""

import os
import site

target_dir = os.path.abspath(__file__ + "/../../build/pymodule/avx2")
sdir = site.getsitepackages()[-1]
pth_path = os.path.join(sdir, "neneshogi_cpp.pth")
print("Creating pth file as " + pth_path)
print("It links to " + target_dir)
print("To uninstall, remove the pth file.")
with open(pth_path, "w") as f:
    f.write(target_dir + "\n")
