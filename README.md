# Python Locks for Safely Using Pytorch On Multiple GPUs
If multiple processes using pytorch on multi-GPUs,
a dead lock will happen on the GPU used by multiple processes.
And the dead lock will cause the GPU become not usable until the whole system reboots.

Thus, I develop this tool to avoid the condition happens.

Normally you just need to call the get_locks function before using Pytorch on GPUs.
```python 2.7
from locks_for_pytorch_on_gpus import acquire_locks, DeviceIsBusyException
locks, msg = acquire_locks()
print msg
try:
    acquire_locks()
except DeviceIsBusyException, e:
    print e

```
Please read the test script as an example.

The code has been tested on python2.7.