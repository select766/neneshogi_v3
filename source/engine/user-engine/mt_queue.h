/*
マルチスレッド対応キュー
https://github.com/juanchopanza/cppblog/blob/master/Concurrency/Queue/Queue.h
を元に、ノンブロッキングpop・複数アイテムの一括投入等DNNに必要な機構を追加
*/
//
// Copyright (c) 2013 Juan Palacios juan.palacios.puyana@gmail.com
// Subject to the BSD 2-Clause License
// - see < http://opensource.org/licenses/BSD-2-Clause>
//

#ifndef CONCURRENT_QUEUE_
#define CONCURRENT_QUEUE_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class MTQueue
{
public:

	T pop()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto val = queue_.front();
		queue_.pop();
		return val;
	}

	bool pop_nb(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		if (queue_.empty())
		{
			return false;
		}
		item = queue_.front();
		queue_.pop();
		return true;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
	}

	size_t pop_batch(T* items, size_t max_size)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		size_t pop_count = 0;
		for (size_t i = 0; i < max_size; i++)
		{
			items[i] = queue_.front();
			queue_.pop();
			pop_count++;
			if (queue_.empty())
			{
				break;
			}
		}

		return pop_count;
	}

	size_t pop_batch_nb(T* items, size_t max_size)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		if (queue_.empty())
		{
			return 0;
		}
		size_t pop_count = 0;
		for (size_t i = 0; i < max_size; i++)
		{
			items[i] = queue_.front();
			queue_.pop();
			pop_count++;
			if (queue_.empty())
			{
				break;
			}
		}

		return pop_count;
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(item);
		mlock.unlock();
		cond_.notify_one();
	}

	void push_batch(const T* items, size_t size)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		for (size_t i = 0; i < size; i++)
		{
			queue_.push(items[i]);
		}
		mlock.unlock();
		cond_.notify_one();
	}
	MTQueue() = default;
	MTQueue(const MTQueue&) = delete;            // disable copying
	MTQueue& operator=(const MTQueue&) = delete; // disable assignment

private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
};

#endif
