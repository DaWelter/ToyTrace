#pragma once

#include <assert.h>
#include <type_traits>
#include <boost/iterator/iterator_facade.hpp>

template<class Node>
class LinkListBase;

template<class Node>
struct CircularLinkList
{
  static_assert(std::is_base_of<LinkListBase<Node>, Node>::value);

  static void append(Node& a_, Node& b_)
  {
    auto* a = static_cast<LinkListBase<Node>*>(&a_);
    auto* b = static_cast<LinkListBase<Node>*>(&b_);
    assert(b->next == b);
    auto* tmp = a->next;
    a->next = &b_;
    b->next = tmp;
  }

  //static Node* prev(const Node& a)
  //{
  //  auto* a = static_cast<LinkListBase<Node> const *>(&a_);
  //  while (a->next != a)
  //  {
  //    a = a->next;
  //  }
  //  return a;
  //}

  static Node* next(const Node& a_)
  {
    auto* a = static_cast<LinkListBase<Node> const *>(&a_);
    return a->next;
  }

  class iterator : public boost::iterator_facade<
    iterator
    , Node
    , boost::forward_traversal_tag>
  {
    Node* cur;
    bool wasIncremented;

    friend class boost::iterator_core_access;

    iterator(Node& a_, bool end)
      : cur{ &a_ }, wasIncremented{ end }
    {}

    friend class CircularLinkList<Node>;

    void increment()
    {
      cur = CircularLinkList<Node>::next(*cur);
      wasIncremented = true;
    }
    bool equal(const iterator &other) const
    {
      return cur == other.cur && wasIncremented == other.wasIncremented;
    }
    Node& dereference() const
    {
      return *cur;
    }
  };

  static iter_pair<iterator> rangeStartingAt(Node &a)
  {
    return {
      { a, false },
      { a, true }
    };
  }
};


template<class Node>
class LinkListBase
{
public:
  friend struct CircularLinkList<Node>;
  LinkListBase()
    : next{ static_cast<Node*>(this) }
  {}
private:
  Node* next = nullptr;
};



