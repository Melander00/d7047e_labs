from scripts.train_attention import main as att
from scripts.train_resnet import main as res
from scripts.train_base import main as base


def main(data_dir = "data"):
    att(data_dir)
    res(data_dir)
    base(data_dir)

if __name__ == "__main__":
    main()
