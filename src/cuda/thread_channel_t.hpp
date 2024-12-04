#pragma once

#include <condition_variable>
#include <deque>

template <class T>
class channel_t
{
private:
  std::deque<T> fifo;
  std::mutex mtx;
  std::condition_variable cvar;

public:
  void send(const T& msg)
  {
    std::unique_lock<std::mutex> lock(mtx);
    fifo.push_back(msg);
    cvar.notify_one();
  }

  T recv()
  {
    std::unique_lock<std::mutex> lock(mtx);
    cvar.wait(lock, [&] { return !fifo.empty(); });
    auto msg = fifo.front();
    fifo.pop_front();
    return msg;
  }
};