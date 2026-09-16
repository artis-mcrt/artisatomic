"""Entry point for `python -m artisatomic`.

`coverage run -m` needs a module name, and the console script makeartisatomicfiles is not one.
"""

from artisatomic.cli import main

if __name__ == "__main__":
    main()
