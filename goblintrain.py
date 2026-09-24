"""The one entry point. ``python goblintrain.py --help`` lists the commands."""
import sys

from goblintrain.cli import main

if __name__ == "__main__":
    sys.exit(main())
