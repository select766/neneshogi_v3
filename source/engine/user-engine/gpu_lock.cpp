#include "../../extra/all.h"
#include <Windows.h>

// GPUロックタイムアウト
// epochからの秒数がこの値以上の時、ロックを解放する。
static atomic<std::chrono::seconds> gpu_lock_timeout(std::chrono::seconds(0));
static std::thread* gpu_lock_thread = nullptr;

// GPUをロックするスレッド。
// 特定のMutexを作成することで、学習プロセスはそれを察知してGPU利用を一旦停止する。
// タイムアウトでMutexを削除する。
static void gpu_lock_thread_main()
{
	HANDLE hMutex = NULL;
	while (true)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
		auto current = chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch());
		if (hMutex && gpu_lock_timeout.load() < current)
		{
			// Mutex削除
			sync_cout << "info string release gpu lock" << sync_endl;
			CloseHandle(hMutex);
			hMutex = NULL;
		}
		else if (!hMutex && gpu_lock_timeout.load() >= current)
		{
			// Mutex作成
			sync_cout << "info string acquire gpu lock" << sync_endl;
			hMutex = CreateMutex(NULL, FALSE, TEXT("NENESHOGI_GPU_LOCK"));
			if (!hMutex)
			{
				sync_cout << "info string FAILED acquire gpu lock" << sync_endl;
			}
		}
	}

}

// GPUをロックするスレッドを開始する。
void gpu_lock_thread_start()
{
	if (!gpu_lock_thread)
	{
		gpu_lock_thread = new std::thread(gpu_lock_thread_main);
		gpu_lock_thread->detach();//プロセス終了時に自動的に終了させる
	}
}

// GPUのロックタイムアウトを延長する。
void gpu_lock_extend()
{
	sync_cout << "info string extending gpu lock" << sync_endl;
	auto next_timeout = chrono::system_clock::now() + chrono::seconds(60);//1手60秒以上はめったにないのでこれぐらいで
	gpu_lock_timeout.store(chrono::duration_cast<chrono::seconds>(next_timeout.time_since_epoch()));
}
