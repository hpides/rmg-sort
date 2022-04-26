#pragma once

#include <future>

struct BucketId {
  BucketId() {
    partition_pass = 0;
    bucket_number = 0;
    predecessor = nullptr;
  }

  BucketId(size_t partition_pass_, size_t bucket_number_) {
    partition_pass = partition_pass_;
    bucket_number = bucket_number_;
    predecessor = nullptr;
  }

  BucketId(size_t partition_pass_, size_t bucket_number_, BucketId* predecessor_) {
    partition_pass = partition_pass_;
    bucket_number = bucket_number_;
    predecessor = predecessor_;
  }

  size_t partition_pass;
  size_t bucket_number;
  BucketId* predecessor;
};

struct compareBucketIds {
  bool operator()(const BucketId& a, const BucketId& b) const {
    if (a.partition_pass == b.partition_pass) {
      return a.bucket_number < b.bucket_number;
    }

    return a.partition_pass < b.partition_pass;
  }
};

template <typename T>
struct ReducedSortingBucket {
  size_t bucket_size;
  size_t bucket_start;
  size_t partition_pass;
  uint32_t msb_dif_position;
  uint32_t bucket_number;

  cub::DoubleBuffer<T> cub_double_buffer;
};

template <typename T>
struct CompareReducedSortingBuckets {
  inline bool operator()(const ReducedSortingBucket<T>& a, const ReducedSortingBucket<T>& b) { return (a.bucket_start < b.bucket_start); }
};

struct LPSpanningBucketFraction {
  int dest_gpu;
  int source_gpu;
  size_t fraction_size;
  size_t source_offset;
  size_t dest_offset;
};