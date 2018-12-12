"""
Mutexを用いて、将棋エンジンが動いている間は学習を止めるユーティリティ。
"""
import win32api
import win32con
import win32event
import pywintypes
import time

MUTEX_ALL_ACCESS = win32con.STANDARD_RIGHTS_REQUIRED | win32con.SYNCHRONIZE | win32con.MUTANT_QUERY_STATE


class MutexStopper:
    def __init__(self, mutex_name="NENESHOGI_GPU_LOCK"):
        self.mutex_name = mutex_name

    def wait(self):
        n_wait_seconds = 0
        while self._is_mutex_exist():
            n_wait_seconds += 1
            time.sleep(1)
        return n_wait_seconds

    def _is_mutex_exist(self):
        try:
            hMutex = win32event.OpenMutex(MUTEX_ALL_ACCESS, False, self.mutex_name)
            win32api.CloseHandle(hMutex)
        except:
            return False
        return True
