from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from part_a import run as a
from part_b import run as b
from part_c import run as c
from part_d import run as d
if __name__=='__main__':
    a(); b(); c(); d()
