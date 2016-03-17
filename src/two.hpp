#pragma once

#include <utility>
#include <stdexcept>

namespace studd
{
    template<class T>
    class two : public std::pair<T, T>
    {
    public:
        using std::pair<T, T>::pair;

        constexpr const T& operator[](std::size_t i) const
        {
            if (i == 0)
            {
                return this->first;
            }
            else if (i == 1)
            {
                return this->second;
            }

            throw std::domain_error("only indices 0 and 1 are valid");
        }

        constexpr T& operator[](std::size_t i)
        {
            if (i == 0)
            {
                return this->first;
            }
            else if (i == 1)
            {
                return this->second;
            }

            throw std::domain_error("only indices 0 and 1 are valid");
        }
    };

    template<class T>
    constexpr two<T> make_two(T&& x, T&& y)
    {
        return two<T>(std::forward<T>(x), std::forward<T>(y));
    }
}