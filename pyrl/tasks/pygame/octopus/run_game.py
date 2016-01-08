#! /usr/bin/env python
import os
from Lake import main

level = os.environ.get('level')
level = int(level) if level else 1

main.main(level)
