#pragma once
#include <mutex>

class ipqueue_meta
{
public:
	int size, read_end, read_begin, write_end, write_begin;
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
	std::mutex _mut;
	void* dataBuf;
	ipqueue_meta metaBuf;
	size_t _size;
	size_t _batch_size;
	size_t item_size;

public:
	bool ok;

	ipqueue(size_t size, size_t batch_size): _size(size), _batch_size(batch_size)
	{
		ok = false;
		metaBuf.init(_size);
		item_size = sizeof(ipqueue_item<T>) + sizeof(T) * (_batch_size - 1);
		size_t map_size = _size * item_size;
		dataBuf = new char[map_size]();

		ok = true;
	}

	~ipqueue()
	{
		if (ok)
		{
			delete[] dataBuf;
		}
	}

	ipqueue_item<T>* begin_read()
	{
		std::lock_guard<std::mutex> lock(_mut);
		int ret = metaBuf.begin_read();
		if (ret < 0)
		{
			return nullptr;
		}
		return reinterpret_cast<ipqueue_item<T>*>(reinterpret_cast<char*>(dataBuf) + item_size * ret);
	}

	void end_read()
	{
		std::lock_guard<std::mutex> lock(_mut);
		metaBuf.end_read();
	}

	ipqueue_item<T>* begin_write()
	{
		std::lock_guard<std::mutex> lock(_mut);
		int ret = metaBuf.begin_write();
		if (ret < 0)
		{
			return nullptr;
		}
		return reinterpret_cast<ipqueue_item<T>*>(reinterpret_cast<char*>(dataBuf) + item_size * ret);
	}

	void end_write()
	{
		std::lock_guard<std::mutex> lock(_mut);
		metaBuf.end_write();
	}

	size_t size() const
	{
		return this->_size;
	}

	size_t batch_size() const
	{
		return this->_batch_size;
	}
};
