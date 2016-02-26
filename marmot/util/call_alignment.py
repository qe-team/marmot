from __future__ import print_function
import sys
from marmot.util.alignments import align_files


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python call_alignment.py src_file tg_file model")
        sys.exit()
    align_files(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[1]+'.align')
