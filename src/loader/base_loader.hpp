#pragma once

#include <string>

template<class Frame, class Calibration>
class base_loader
{
public:
    base_loader(const Calibration& c) : c(c) { }

    virtual Frame operator[](size_t i) = 0;
    virtual Calibration get_calibration() const { return c; }

protected:
    Calibration c;
};
