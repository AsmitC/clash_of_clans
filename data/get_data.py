import subprocess
import sys

# Install roboflow
subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])

from roboflow import Roboflow

# Call for dataset
rf = Roboflow(api_key="my_key")
project = rf.workspace("find-this-base").project("clash-of-clans-vop4y")
version = project.version(5)
dataset = version.download("yolov5")

# Call for preprocessed dataset
rf = Roboflow(api_key="my_key")
project = rf.workspace("asmit-chakraborty").project("clash_of_clans-7n0sd")
version = project.version(1)
dataset = version.download("yolov5")

