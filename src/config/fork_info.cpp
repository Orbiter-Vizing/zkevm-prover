#include <array>
#include "fork_info.hpp"

using namespace std;

#define NO_COUNTERS_MULTIPLIER uint64_t(256)

array<ForkInfo, 12> forkInfo = {
//            ID   PID  Nbits        N N_NoCounters
    ForkInfo(  0,    0,   0,        0,          0 ),
    ForkInfo(  1,    1,  23,  8388608,  NO_COUNTERS_MULTIPLIER * 8388608 ),
    ForkInfo(  2,    2,  23,  8388608,  NO_COUNTERS_MULTIPLIER * 8388608 ),
    ForkInfo(  3,    3,  23,  8388608,  NO_COUNTERS_MULTIPLIER * 8388608 ),
    ForkInfo(  4,    4,  23,  8388608,  NO_COUNTERS_MULTIPLIER * 8388608 ),
    ForkInfo(  5,    5,  23,  8388608,  NO_COUNTERS_MULTIPLIER * 8388608 ),
    ForkInfo(  6,    6,  25,  33554432,  NO_COUNTERS_MULTIPLIER * 33554432 )
};

bool getForkInfo (uint64_t forkID, ForkInfo &_forkInfo)
{
    if (forkID == 0) return false;
    if (forkID >= forkInfo.size()) return false;
    _forkInfo = forkInfo[forkID];
    return true;
}

uint64_t getForkN (uint64_t forkID)
{
    if (forkID == 0) return 0;
    if (forkID >= forkInfo.size()) return 0;
    return forkInfo[forkID].N;
}
