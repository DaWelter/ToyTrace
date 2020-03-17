#pragma once


#ifdef HAVE_JSON
namespace rapidjson
{

class CrtAllocator;

template <typename BaseAllocator>
class MemoryPoolAllocator;

template <typename Encoding, typename Allocator>
class GenericValue;

template <typename Encoding, typename Allocator, typename StackAllocator>
class GenericDocument;

template<typename CharType>
struct UTF8;


typedef GenericDocument<UTF8<char>, MemoryPoolAllocator<CrtAllocator>, CrtAllocator> Document;
typedef GenericValue<UTF8<char>, MemoryPoolAllocator<CrtAllocator>> Value;
} // rapidjson
#endif
