#pragma once

namespace studd
{
    struct pair_hash {
    public:
        template <typename T, typename U>
        std::size_t operator()(const std::pair<T, U> &x) const
        {
            return std::hash<T>()(x.first) ^ (std::hash<U>()(x.second) << 16);
        }
    };
}