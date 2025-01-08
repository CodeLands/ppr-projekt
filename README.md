```
git clone https://github.com/CodaLands/ppr-projekt integration
```

```
sudo dnf groupinstall "Development Tools"
sudo dnf install gcc gcc-c++
```

```
sudo dnf install openmpi openmpi-devel
```

```
MPICC=mpicc pip install mpi4py --no-cache-dir
```

```
sudo pip install -r requirements.txt
```

```
add this to .bashenv: export PATH="/usr/lib64/openmpi/bin:$PATH"
```

```
mpirun -np 2 --host pckill3r@192.168.175.34,zorko@192.168.175.31 python3 integration/distributed_image_augmentation.py
```
