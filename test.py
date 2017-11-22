from __init__ import acquire_locks, release_locks, DEFAULT_LOG_DIR, DeviceIsBusyException
import os


def test():
    locks, msg = acquire_locks([0, 1, 2, 3])
    print msg
    with open(DEFAULT_LOG_DIR + "/gpu_3.pid") as fd:
        print fd.read(), "==", os.getpid()

    release_locks(locks)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    _, msg = acquire_locks()
    print msg

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,0"
    _, msg = acquire_locks()
    print msg

    _, msg = acquire_locks([0])
    print msg

    _, msg = acquire_locks([])
    print msg

    try:
        del os.environ["CUDA_VISIBLE_DEVICES"]
        acquire_locks()

    except DeviceIsBusyException, e:
        print e


if __name__ == '__main__':
    test()
