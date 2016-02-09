#pragma once

#include <unordered_map>
#include <array>
#include <functional>

namespace studd
{
    template<class Key,
             class T,
             class Hash = std::hash<Key>,
             class KeyEqual = std::equal_to<Key>,
             class InternalMap = std::unordered_map<Key, T>,
             class Allocator = std::allocator<std::pair<const Key, T>>>
    class dynamic_map
    {
    public:
        dynamic_map(std::function<T(const Key&)> producer) : producer(producer) { };

        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<const Key, T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = typename std::allocator_traits<Allocator>::pointer;
        using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;

        T& operator[](const Key& k)
        {
            auto it = mapping.find(k);
            if (it == mapping.end())
            {
                num_active++;
                auto new_it = mapping.insert(it, std::make_pair(k, producer(k)));
                return mapping[k];
            }
            else
            {
                return it->second;
            }
        }

        size_type size() const
        {
            return num_active;
        }

        bool empty() const
        {
            return num_active == 0;
        }

        void clear()
        {
            num_active = 0;
            mapping.clear();
        }

    private:
        InternalMap mapping;
        std::size_t num_active = 0;
        std::function<T(const Key&)> producer;
    };
}