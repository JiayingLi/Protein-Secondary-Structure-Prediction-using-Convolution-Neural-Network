from __future__ import print_function
import sys

def warning(*objs):
    print(*objs, file=sys.stderr)

if len(sys.argv) <= 2:
    print('usage: python %s filename size' % sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]
size = int(sys.argv[2])

with open(filename, 'r') as f:
    total_num = int(f.readline())
    print(size)
    for __ in range(size):
        n = int(f.readline())
        print(n)
        for __ in range(n):
            print(f.readline()[:-1])
        print(f.readline()[:-1])

    warning(total_num - size)
    for __ in range(total_num - size):
        n = int(f.readline())
        warning(n)
        for __ in range(n):
            warning(f.readline()[:-1])
        warning(f.readline()[:-1])

