import sys
from pathlib import Path

# The package lives at <root>/cued_recall/ and the tests at <root>/tests/, so
# pytest's own sys.path insertion (the test file's own directory) is not enough
# to `import cued_recall`. Add the project root explicitly rather than requiring
# an editable install -- this project is run from a checkout, not from site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
