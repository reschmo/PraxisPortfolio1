# Ermöglicht `import src...` in den Tests, egal von wo pytest gestartet wird.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
