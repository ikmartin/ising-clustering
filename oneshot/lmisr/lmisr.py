#!/bin/sh

################################################################################
## Shell Interpreter
################################################################################

""":"

for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "ERROR: Recognized version of Python interpreter not found" 

exit 1

":"""

################################################################################

from sys import stderr

# Create c_types to point to isingLPA functions in the shared object

from ctypes import cdll, c_bool, c_double, c_int, POINTER

try:
    lib = cdll.LoadLibrary('isingLPA.so')
except OSError:
    stderr.write(f'ERROR: Shared object "isingLPA.so" not found\n')
    exit()

c_free_ptr = lib.free_ptr
c_lmisr = lib.lmisr
c_lmisr.restype = POINTER(c_double)
