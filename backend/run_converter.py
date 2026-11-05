
import numpy as np
import runpy
import sys



# Jalankan converter dengan argumen terminal
sys.argv = [
    "tensorflowjs_converter",
    *sys.argv[1:]
]

runpy.run_module(
    "tensorflowjs.converters.converter",
    run_name="__main__"
)
