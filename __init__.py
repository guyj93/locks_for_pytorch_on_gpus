import os
import socket
import atexit

DEFAULT_LOG_DIR = '/tmp/pytorch-on-gpus-log'


class DeviceIsBusyException(Exception):
    def __init__(self, device_id, log_dir=DEFAULT_LOG_DIR):
        self.device_id = device_id
        self.log_dir = log_dir

    def __str__(self):
        return 'Device {} is busy! Read the pids in log directory "{}" to find which process is using the devices.' \
            .format(self.device_id, self.log_dir)


def _acquire_socket_lock(lock_name):
    # Without holding a reference to our socket somewhere it gets garbage
    # collected when the function exits
    _lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    _lock_socket.bind('\0' + lock_name)
    return _lock_socket


def _release_socket_lock(lock):
    lock.close()


def _acquire_gpu_device_lock(device_id):
    return _acquire_socket_lock("UsingPytorchOnGpu{}".format(device_id))


def release_locks(device_lock_list):
    """Try to release the locks
    """
    # print "releasing", device_lock_list
    for device_lock in device_lock_list:
        try:
            _release_socket_lock(device_lock)
        except:
            pass


def _get_cuda_visible_devices():
    devices_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices_str is None:
        return None
    device_id_list = []
    for device_id_str in devices_str.strip().split(','):
        if devices_str == '':
            continue
        device_id_list.append(int(device_id_str))
    return device_id_list


def acquire_locks(device_id_list=None, num_all_devices=16, log_dir=DEFAULT_LOG_DIR):
    """Try to acquire locks for using pytorch on GPUs
        If you use a single gpu, these is no need to get a lock, thus the function will return no locks
        The locks will automatically release when the process exits, also you can release them manually.

        device_id_list: list of device id to use
            If device_id_list is not given, the function will use environment variable CUDA_VISIBLE_DEVICES instead.
            If both device_id_list is not given and env CUDA_VISIBLE_DEVICES is not set,
            the function will assume you are going to use all GPUs, and the device_id_list will be range(num_all_devices).
        num_all_devices: number of devices
            Only be used when both both device_id_list is not given and env CUDA_VISIBLE_DEVICES is not set.
            The value should be larger than the devices you have. We assume you have less than 16 devices.
            Give an accurate number will help to get the least locks.
        log_dir: a dir to log where to store the pid of this process

        return: a list of locks
    """
    pid = os.getpid()
    if device_id_list is None:
        device_id_list = _get_cuda_visible_devices()
        if device_id_list is None:
            device_id_list = range(num_all_devices)
    num_device = len(device_id_list)
    device_lock_list = []
    if num_device == 1:
        msg = "Info: It is safe for using only one GPU."
    elif num_device == 0:
        msg = "Warning: No GPU device is given!"
    else:
        try:
            for device_id in device_id_list:
                if str(device_id) == "":
                    continue
                device_lock = _acquire_gpu_device_lock(device_id)
                device_lock_list.append(device_lock)
        except socket.error:
            release_locks(device_lock_list)
            raise DeviceIsBusyException(device_id, log_dir)
        atexit.register(release_locks, device_lock_list)
        msg = "Info: Success to get device {}.".format(device_id_list)
        if log_dir is not None:
            try:
                os.makedirs(log_dir)
                os.chmod(log_dir, 0o777)
            except:
                pass
            for device_id in device_id_list:
                device_log_path = os.path.join(log_dir, "gpu_{}.pid".format(device_id))
                with open(device_log_path, "w") as fd:
                    fd.write(str(pid))
                os.chmod(device_log_path, 0o666)
    return device_lock_list, msg


