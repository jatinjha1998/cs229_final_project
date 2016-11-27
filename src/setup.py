import os

log_path = "../logs/"
modules  = ""

# scipy modules
modules = modules + " " + "numpy scipy matplotlib ipython jupyter sympy nose"

# pandas modules
modules = modules + " " + "pandas pandas-datareader"

# machine learning modules
modules = modules + " " + "scikit-learn keras"

# Split module into list
modules = modules.split(" ")

for module in modules:
    # Skip empty module names
    if module == "":
        continue

    # Overwrite previous log
    log = log_path + module + ".log"
    if os.path.isfile(log):
        os.remove(log)

    # Installing module, writing to log
    cmd = "pip install --user " + module + " > " + log
    os.system(cmd)

# Tensorflow
cmd = "pip install --user --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl > " + \
    log_path + "tensorflow.log"
os.system(cmd)
