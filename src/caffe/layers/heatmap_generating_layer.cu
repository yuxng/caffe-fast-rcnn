// ------------------------------------------------------------------
// Subcategory CNN
// Copyright (c) 2015 Stanford CVGL
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/roi_generating_layers.hpp"

using std::cout;

namespace caffe {

template <typename Dtype>
__global__ void HeatmapGeneratingForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height, const int width, Dtype* top_data, int* argmax_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    // (n, 0, h, w) is an element in the output
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;

    // compute the max prob
    int bottom_index_start = n * channels * height * width + h * width + w;
    Dtype max_value = Dtype(-1);
    int max_id = 0;
    for(int c = 1; c < channels; c++)
    {
      int bottom_index = bottom_index_start + c * height * width;
      Dtype value = bottom_data[bottom_index];
      if(value > max_value)
      {
        max_value = value;
        max_id = c;
      }
    }

    // store the max prob and id
    top_data[index] = max_value;
    argmax_data[index] = max_id;
  }
}

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();

  clock_t time_begin = clock();
  // NOLINT_NEXT_LINE(whitespace/operators)
  HeatmapGeneratingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_, height_, width_, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
  clock_t time_end = clock();
  double elapsed_secs = double(time_end - time_begin) / CLOCKS_PER_SEC;
  cout << "Compute heatmap gpu: " << elapsed_secs << " second\n";
}

template <typename Dtype>
__global__ void HeatmapGeneratingBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int channels, const int height, const int width, Dtype* bottom_diff)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int top_index = n * height * width + h * width + w;
    if(c == argmax_data[top_index])
    {
      bottom_diff[index] = top_diff[top_index];
    }
  }
}

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (!propagate_down[0])
    return;

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();

  clock_t time_begin = clock();
  // NOLINT_NEXT_LINE(whitespace/operators)
  HeatmapGeneratingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, channels_, height_, width_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
  clock_t time_end = clock();
  double elapsed_secs = double(time_end - time_begin) / CLOCKS_PER_SEC;
  cout << "Compute heatmap gradient gpu: " << elapsed_secs << " second\n";
}

INSTANTIATE_LAYER_GPU_FUNCS(HeatmapGeneratingLayer);

}  // namespace caffe
