#pragma once

#include <map>
#include <array>
#include <functional>

namespace studd
{
    template<class Key,
             class T,
             std::size_t N,
             class Hash = std::hash<Key>,
             class KeyEqual = std::equal_to<Key>,
             class InternalMap = std::map<Key, std::size_t>,
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
                if (num_active == N)
                {
                    it = find_by_index(index);
                    mapping.erase(it);
                }
                else
                {
                    num_active++;
                }
                auto i = index;
                auto new_it = mapping.emplace(k, i);
                storage[i] = producer(k);
                index = (index + 1) % N;
                return storage[i];
            }
            else
            {
                return storage[it->second];
            }
        }

        size_type size() const
        {
            return N;
        }

        void clear()
        {
            num_active = 0;
            index = 0;
            mapping.clear();
        }

    private:
        InternalMap mapping;
        std::array<T, N> storage;
        std::size_t num_active = 0;
        std::size_t index = 0;
        std::function<T(const Key&)> producer;

        typename InternalMap::iterator find_by_index(std::size_t i)
        {
            for (typename InternalMap::iterator it = mapping.begin();
                 it != mapping.end();
                 ++it)
            {
                if (it->second == i)
                {
                    return it;
                }
            }
            return mapping.end();
        }
    };
}