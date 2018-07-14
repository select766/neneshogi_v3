"""
Register neneshogi C++ python module to import path
Copy __init__.py[i] from avx2 directory to specified arch
"""

import os
import sys
import site
import distutils.file_util

arch = sys.argv[1] if len(sys.argv) == 2 else "avx2"

target_dir = os.path.abspath(__file__ + "/../../build/pymodule/" + arch)
init_src_dir = os.path.abspath(__file__ + "/../../build/pymodule/avx2")
pyd_path = os.path.join(target_dir, "neneshogi_cpp.pyd")
if not os.path.exists(pyd_path):
    print(pyd_path + " does not exist. Abort.")
    sys.exit(1)

if arch != "avx2":
    for fn in ["__init__.py", "__init__.pyi"]:
        distutils.file_util.copy_file(os.path.join(init_src_dir, fn),
                                      os.path.join(target_dir, fn),
                                      verbose=True)
sdir = site.getsitepackages()[-1]
pth_path = os.path.join(sdir, "neneshogi_cpp.pth")
print("Creating pth file as " + pth_path)
print("It links to " + target_dir)
print("To uninstall, remove the pth file.")
with open(pth_path, "w") as f:
    f.write(target_dir + "\n")
