import sys
import time
print(f"{sys.argv[1]} {sys.argv[2]}")
for i in range(5):
    i = i + 1
time.sleep(2)
print(f"Done with {sys.argv[1]}!")

time.sleep(2)
print(f"Fully Done with {sys.argv[1]}!")