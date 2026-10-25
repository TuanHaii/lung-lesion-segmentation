import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from train import main, train_one_epoch, validate, build_model

if __name__ == "__main__":
    main()
