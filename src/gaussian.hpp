#pragma once

#include <iostream>

struct gaussian
{
    gaussian(float mean, float variance)
        : mean(mean), variance(variance) { };

    friend gaussian operator^(const gaussian& lhs, const gaussian& rhs)
    {
        if (rhs.variance < 0)
        {
            return lhs;
        }
        if (lhs.variance < 0)
        {
            return rhs;
        }

        if (std::abs(lhs.mean - rhs.mean) > lhs.variance * 2)
        {
            if (lhs.mean > rhs.mean)
            {
                return lhs;
            }
            else
            {
                return rhs;
            }
        }

        auto mean = (lhs.variance * rhs.mean + rhs.variance * lhs.mean)
                    / (lhs.variance + rhs.variance);
        auto variance = (lhs.variance * rhs.variance) / (lhs.variance + rhs.variance);
        return gaussian(mean, variance);
    }

    friend std::ostream& operator<< (std::ostream& stream, const gaussian& self) {
        stream << self.mean << " " << self.variance;
        return stream;
    }

    float mean;
    float variance;
};
