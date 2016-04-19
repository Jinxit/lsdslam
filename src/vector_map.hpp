#pragma once

#include <unordered_map>
#include <vector>
#include <exception>

template<class Key, class T, class Hash = std::hash<Key>>
class vector_map
{
public:
    vector_map() { };

    const std::pair<Key, T>& operator[](size_t i) const
    {
        return elements[i];
    }

    std::pair<Key, T>& operator[](size_t i)
    {
        invalidated = true;
        return elements[i];
    }

    const T& operator()(const Key& k) const
    {
        if (invalidated)
            throw std::invalid_argument("vector_map has been invalidated");

        return elements[indices.at(k)];
    }

    T& operator()(const Key& k)
    {
        if (invalidated)
            throw std::invalid_argument("vector_map has been invalidated");

        invalidated = true;
        return elements[indices.at(k)];
    }

    void push_back(const Key& k, const T& v)
    {
        elements.emplace_back(k, v);
        indices[k] = elements.size() - 1;
    }

    typename std::vector<std::pair<Key, T>>::iterator begin() { return elements.begin(); }
    typename std::vector<std::pair<Key, T>>::const_iterator begin() const { return elements.begin(); }
    typename std::vector<std::pair<Key, T>>::iterator end() { return elements.end(); }
    typename std::vector<std::pair<Key, T>>::const_iterator end() const { return elements.end(); }
    size_t size() const { return elements.size(); }

private:
    std::vector<std::pair<Key, T>> elements;
    std::unordered_map<Key, size_t, Hash> indices;

    bool invalidated = false;
};