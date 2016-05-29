#pragma once

namespace studd
{
    template<class T1>
    struct static_cast_func
    {
        template <class T2>
        T1 operator()(const T2& x) const
        {
            return static_cast<T1>(x);
        }
    };
}