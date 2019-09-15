#pragma once

#include <assert.h>
#include <type_traits>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/none_t.hpp>

// Use CRTP
template<class Node, class Tag = boost::none_t>
class LinkListBase;

// This struct only collects operations on the list nodes.
// It is not meant to be derived from or composed into client objects.
template<class Node, class Tag = boost::none_t>
struct CircularLinkList
{
  static_assert(std::is_base_of<LinkListBase<Node, Tag>, Node>::value);

  static void append(Node& a_, Node& b_)
  {
    // The cast selects which of the potentially many LinkListBase's I want to use.
    auto* a = static_cast<LinkListBase<Node,Tag>*>(&a_);
    auto* b = static_cast<LinkListBase<Node,Tag>*>(&b_);
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
    auto* a = static_cast<LinkListBase<Node, Tag> const *>(&a_);
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

    friend struct CircularLinkList<Node, Tag>;

    void increment()
    {
      cur = CircularLinkList<Node, Tag>::next(*cur);
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


template<class Node, class Tag>
class LinkListBase
{
public:
  friend struct CircularLinkList<Node, Tag>;
  LinkListBase()
    : next{ static_cast<Node*>(this) }
  {}

  LinkListBase(const LinkListBase &) = delete;
  LinkListBase(LinkListBase &&) = delete;

private:
  Node* next = nullptr;
};



