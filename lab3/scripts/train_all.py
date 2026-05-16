import os
os.environ['TORCH_HOME'] = '/home/d7047e_labs/torch_cache'

from scripts.train_attention import main as att
from scripts.train_resnet import main as res
from scripts.train_base import main as base


def main(data_dir = "Data"):
    att(data_dir)
    res(data_dir)
    base(data_dir)

if __name__ == "__main__":
    main()
