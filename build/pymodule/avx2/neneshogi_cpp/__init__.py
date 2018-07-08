
import os
os.environ["PATH"] += os.pathsep + os.path.dirname(__file__)

from .neneshogi_cpp import *
