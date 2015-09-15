// ------------------------------------------------------------------
// Subcategory CNN
// Copyright (c) 2015 Stanford CVGL
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/roi_generating_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FeatureExtrapolatingForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height, const int width, Dtype* top_data, 
    const int* flags, const int* mapping, const double* factors, int num_scale, int num_scale_base, int channels_trace, double* trace_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, h, w) is an element in the output
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int index_image = n / num_scale;
    int index_scale = n % num_scale;
    // flag for approximation or not
    int flag = flags[index_scale];
    // which base scale to use
    int index_scale_base = mapping[index_scale];
    // rescaling factor
    double factor = factors[index_scale];
    // bottom batch image
    int index_batch = index_image * num_scale_base + index_scale_base;
    const Dtype* batch_data = bottom_data + index_batch * channels * height * width + c * height * width;

    if(flag == 1) // no approximation
    {
      top_data[index] = batch_data[h * width + w];
      // set tracing info
      if(c == 0)
      {
        for(int i = 0; i < channels_trace / 2; i++)
        {
          trace_data[n * channels_trace * height * width + 2 * i * height * width + h * width + w] = index_batch * channels * height * width + h * width + w;
          trace_data[n * channels_trace * height * width + (2 * i + 1) * height * width + h * width + w] = 0.25;
        }
      }
    }
    else
    {
      // bilinear interpolation
      double xp = w / factor;
      double yp = h / factor;
      double cx[2], cy[2], ux, uy;
      int xi, yi, dx, dy, i;
      Dtype val;
      if(xp >= 0 && xp < width && yp >= 0 && yp < height)
      {
        xi = (int)floor(xp); 
        yi = (int)floor(yp);
        ux = xp - (double)xi;
        uy = yp - (double)yi;
        cx[0] = ux;
        cx[1] = 1 - ux;
        cy[0] = uy;
        cy[1] = 1 - uy;

        val = 0;
        i = 0;
        for(dx = 0; dx <= 1; dx++)
        {
          for(dy = 0; dy <= 1; dy++)
          {
            if(xi+dx >= 0 && xi+dx < width && yi+dy >= 0 && yi+dy < height)
            {
              val += cx[1-dx] * cy[1-dy] * batch_data[(yi+dy) * width + (xi+dx)];
              if(c == 0)
              {
                trace_data[n * channels_trace * height * width + 2 * i * height * width + h * width + w] = index_batch * channels * height * width + (yi+dy) * width + (xi+dx);
                trace_data[n * channels_trace * height * width + (2 * i + 1) * height * width + h * width + w] = cx[1-dx] * cy[1-dy];
              }
            }
            else
            {
              if(c == 0)
              {
                trace_data[n * channels_trace * height * width + 2 * i * height * width + h * width + w] = -1;
                trace_data[n * channels_trace * height * width + (2 * i + 1) * height * width + h * width + w] = 0;
              }
            }
            i++;
          }
        }
        top_data[index] = val;
      }
      else
      {
        // set tracing info
        if(c == 0)
        {
          for(int i = 0; i < channels_trace / 2; i++)
          {
            trace_data[n * channels_trace * height * width + 2 * i * height * width + h * width + w] = -1;
            trace_data[n * channels_trace * height * width + (2 * i + 1) * height * width + h * width + w] = 0;
          }
        }
      }
    }
  }
}

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  caffe_gpu_set(count, Dtype(0.), top_data);
  double* trace_data = trace_.mutable_gpu_data();

  const int* flags = is_real_scales_.gpu_data();
  const int* mapping = which_base_scales_.gpu_data();
  const double* factors = rescaling_factors_.gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  FeatureExtrapolatingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_, height_, width_, top_data, flags, mapping, factors, num_scale_, num_scale_base_, channels_trace_, trace_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void FeatureExtrapolatingBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height, const int width, Dtype* bottom_diff, const int* mapping, const double* factors,
    int num_scale, int num_scale_base, int channels_trace, const double* trace_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    // (n, c, h, w) coords in bottom diff
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int index_image = n / num_scale_base;
    int index_scale_base = n % num_scale_base;

    Dtype val = 0;
    for(int i = 0; i < num_scale; i++)
    {
      if(mapping[i] == index_scale_base)
      {
        int index_batch = index_image * num_scale + i;
        double factor = factors[i];
        double xp = w * factor;
        double yp = h * factor;
        int xi = (int)floor(xp); 
        int yi = (int)floor(yp);
        
        for(int dx = -2; dx <= 2; dx++)
        {
          for(int dy = -2; dy <= 2; dy++)
          {
            if(xi+dx >= 0 && xi+dx < width && yi+dy >= 0 && yi+dy < height)
            {
              for(int j = 0; j < channels_trace / 2; j++)
              {
                int index_trace = int(trace_data[index_batch * channels_trace * height * width + 2 * j * height * width + (yi+dy) * width + (xi+dx)]);
                double weight_trace = trace_data[index_batch * channels_trace * height * width + (2 * j + 1) * height * width + (yi+dy) * width + (xi+dx)];
                if(index_trace == n * channels * height * width + h * width + w)
                  val += weight_trace * top_diff[index_batch * channels * height * width + c * height * width + (yi+dy) * width + (xi+dx)];
              }
            }
          }
        }
      }
    }
    // assign value
    bottom_diff[index] = val;
  }
}

template <typename Dtype>
void FeatureExtrapolatingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (!propagate_down[0])
    return;

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  const double* trace_data = trace_.gpu_data();
  const int* flags = is_real_scales_.gpu_data();
  const int* mapping = which_base_scales_.gpu_data();
  const double* factors = rescaling_factors_.gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  FeatureExtrapolatingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, channels_, height_, width_, bottom_diff, mapping, factors, num_scale_, num_scale_base_, channels_trace_, trace_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(FeatureExtrapolatingLayer);

}  // namespace caffe
