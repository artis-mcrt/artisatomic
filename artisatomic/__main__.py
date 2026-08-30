"""Entry point for `python -m artisatomic`, which `coverage run -m` can name and the console script cannot."""

from artisatomic.cli import main

if __name__ == "__main__":
    main()
