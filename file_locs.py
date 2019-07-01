import socket

name = socket.gethostname()
import pandas as pd
import math



data_dir="/work/drl21/"


if name.startswith("dcc"):
    mm_data="/dscrhome/drl21/mmdetection/data/"
    mmdl_dir=mm_data+"deeplesion/"
    image_dir=mmdl_dir+"Images_png/"
    csv_dir=mmdl_dir
else:
    raise AssertionError("unrecognized host")