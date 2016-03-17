#pragma once

namespace studd
{
    template<class T>
    inline constexpr T square(T value)
    {
        return value * value;
    }
}