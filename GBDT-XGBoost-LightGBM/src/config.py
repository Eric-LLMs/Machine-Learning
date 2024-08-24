import os
from collections import namedtuple

root = os.getcwd()
params_processing = namedtuple(
  "Params",
  [
    "root",
    "data_dir",
    "output_dir"
  ])


def create_params():
    return params_processing(
      root=root,
      data_dir=os.path.join(root, 'data'),
      output_dir=os.path.join(root, 'output')
    )

# print (create_params().data_dir)
# print (create_params().root)
# print (create_params().output_dir)
