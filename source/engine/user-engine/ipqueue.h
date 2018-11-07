#pragma once
#include <Windows.h>
#include <string>
#include <locale>
#include <codecvt>

class ipqueue_meta
{
public:
	// batch_sizeは初期化値の共有用
	int size, batch_size, read_end, read_begin, write_end, write_begin;
	int watcher_count;// CreateFileMappingで零初期化されるはず https://msdn.microsoft.com/ja-jp/library/windows/desktop/aa366537(v=vs.85).aspx
	// The initial contents of the pages in a file mapping object backed by the operating system paging file are 0 (zero).
	bool writing, reading;

	void init(int size)
	{
		this->size = size;
		read_end = 0;
		read_begin = 0;
		write_end = 0;
		write_begin = 0;
		writing = false;
		reading = false;
	}

	int begin_read()
	{
		if (reading)
		{
			return -1;
		}
		if (write_end != read_begin)
		{
			int cur = read_begin;
			read_begin = (read_begin + 1) % size;
			reading = true;
			return cur;
		}
		else
		{
			return -1;
		}
	}

	void end_read()
	{
		reading = false;
		read_end = (read_end + 1) % size;
	}

	int begin_write()
	{
		if (writing)
		{
			return -1;
		}
		int cur = write_begin;
		int next = (cur + 1) % size;
		if (next != read_end)
		{
			writing = true;
			write_begin = next;
			return cur;
		}
		else
		{
			return -1;
		}
	}

	void end_write()
	{
		writing = false;
		write_end = (write_end + 1) % size;
	}
};

class lock_mutex
{
	HANDLE hMutex;

public:
	lock_mutex(HANDLE hMutex)
	{
		this->hMutex = hMutex;
		WaitForSingleObject(hMutex, INFINITE);
	}

	~lock_mutex()
	{
		ReleaseMutex(hMutex);
	}
};

template<typename T>
class ipqueue_item
{
public:
	int count;
	T elements[1];
};

template<typename T>
class ipqueue
{
	HANDLE hMapFileMeta;
	HANDLE hMapFileData;
	void* hMapBuf;
	void* dataBuf;
	ipqueue_meta* metaBuf;
	HANDLE hMutex;
	size_t _size;
	size_t _batch_size;
	size_t item_size;

public:
	bool ok;

	ipqueue(size_t size, size_t batch_size, const std::string& name, bool initialize)
	{
		ok = false;
		// utf-8で名前を受け取り、utf-16に変換
		std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
		std::wstring namew = cv.from_bytes(name);
		std::wstring map_name_meta = namew + L"_map_meta";
		std::wstring map_name_data = namew + L"_map_data";
		std::wstring mutex_name = namew + L"_mutex";
		hMapFileMeta = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(ipqueue_meta), map_name_meta.c_str());
		if (hMapFileMeta ==NULL)
		{
			return;
		}
		metaBuf = reinterpret_cast<ipqueue_meta*>(MapViewOfFile(hMapFileMeta, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(ipqueue_meta)));
		if (metaBuf == NULL)
		{
			return;
		}
		if (initialize)
		{
			hMutex = CreateMutex(NULL, FALSE, mutex_name.c_str());
			if (hMutex == NULL)
			{
				return;
			}
			metaBuf->init(size);
			metaBuf->batch_size = batch_size;
			this->_size = size;
			this->_batch_size = batch_size;
		}
		else
		{
			hMutex = OpenMutex(MUTEX_ALL_ACCESS, FALSE, mutex_name.c_str());
			if (hMutex == NULL)
			{
				return;
			}
			this->_size = metaBuf->size;
			this->_batch_size = metaBuf->batch_size;
		}

		item_size = sizeof(ipqueue_item<T>) + sizeof(T) * (_batch_size - 1);
		size_t map_size = _size * item_size;
		hMapFileData = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, map_size, map_name_data.c_str());
		if (hMapFileData == NULL)
		{
			return;
		}
		dataBuf = reinterpret_cast<ipqueue_item<T>*>(MapViewOfFile(hMapFileData, FILE_MAP_ALL_ACCESS, 0, 0, map_size));
		if (dataBuf == NULL)
		{
			return;
		}

		ok = true;
	}

	~ipqueue()
	{
		if (ok)
		{
			UnmapViewOfFile(dataBuf);
			CloseHandle(hMapFileData);
			UnmapViewOfFile(metaBuf);
			CloseHandle(hMapFileMeta);
			CloseHandle(hMutex);
		}
	}

	ipqueue_item<T>* begin_read()
	{
		lock_mutex m(hMutex);
		int ret = metaBuf->begin_read();
		if (ret < 0)
		{
			return nullptr;
		}
		return reinterpret_cast<ipqueue_item<T>*>(reinterpret_cast<char*>(dataBuf) + item_size * ret);
	}

	void end_read()
	{
		lock_mutex m(hMutex);
		metaBuf->end_read();
	}

	ipqueue_item<T>* begin_write()
	{
		lock_mutex m(hMutex);
		int ret = metaBuf->begin_write();
		if (ret < 0)
		{
			return nullptr;
		}
		return reinterpret_cast<ipqueue_item<T>*>(reinterpret_cast<char*>(dataBuf) + item_size * ret);
	}

	void end_write()
	{
		lock_mutex m(hMutex);
		metaBuf->end_write();
	}

	size_t size()
	{
		return this->_size;
	}

	size_t batch_size()
	{
		return this->_batch_size;
	}

	int increment_watcher_count()
	{
		lock_mutex m(hMutex);
		return ++(metaBuf->watcher_count);
	}

	int get_watcher_count()
	{
		lock_mutex m(hMutex);
		return metaBuf->watcher_count;
	}
};
