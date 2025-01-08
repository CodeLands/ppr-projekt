
```
export PATH="/usr/lib64/openmpi/bin:$PATH"
```

```
mpirun -np 2 --host pckill3r@192.168.175.34,zorko@192.168.175.31 python3 integration/distributed_image_augmentation.py
```
