import sys
import hashlib
import pathlib

for p in map(pathlib.Path, sys.argv[1:]):
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(p.name, p.stat().st_size, h)
