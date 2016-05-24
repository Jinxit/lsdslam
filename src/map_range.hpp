#pragma once

namespace studd
{
    template<class T>
    T map_range(T x, T a1, T a2, T b1, T b2)
    {
        return b1 + (x - a1) * (b2 - b1) / (a2 - a1);
    }
}