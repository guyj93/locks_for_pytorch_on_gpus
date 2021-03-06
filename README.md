# Python Locks for Safely Using Pytorch On Multiple GPUs
If there are multiple processes using pytorch on multi-GPUs,
a dead lock will happen on the GPU used by multiple processes.
And the dead lock will cause the GPU become not available until the OS reboots.

Thus, I develop this tool to avoid the condition happens.

## Requirements
- Linux (The abstract sockets are used to create locks)
- Python2.7

## Installation
As an temporary solution with no dependency, I don't pack the module.

Please download the code and copy the directory "locks_for_pytorch_on_gpus" into your project!

## Usage
Normally you just need to call the acquire_locks function before using Pytorch on GPUs.
For example, open an python console and input:
```python
from locks_for_pytorch_on_gpus import acquire_locks
locks, msg = acquire_locks()
print msg
```
Then, open another python console and input:
```python
from locks_for_pytorch_on_gpus import acquire_locks, DeviceIsBusyException
try:
    acquire_locks()
except DeviceIsBusyException, e:
    print e

```
