from typing import List

import torch



def main() -> None:
    # Check for cuda and set device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    main()
