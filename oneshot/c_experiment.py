from ctypes import c_int, CDLL

lib = CDLL('./lmisr/test.so')

c_test = lib.test
c_test.restype = c_int

result = c_test(
    c_int(2),
    c_int(3),
)

print(result)
