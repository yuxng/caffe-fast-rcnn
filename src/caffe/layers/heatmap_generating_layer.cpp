// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 CVGL Stanford
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

# include <cfloat>
# include <ctime>
# include <stdio.h>

#include "caffe/roi_generating_layers.hpp"

using std::cout;

namespace caffe {

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // heatmap
  top[0]->Reshape(num_, 1, height_, width_);
  max_idx_.Reshape(num_, 1, height_, width_);
}

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  Dtype* heatmap = top[0]->mutable_cpu_data();
  int* argmax_data = max_idx_.mutable_cpu_data();

  // compute heatmap
  clock_t time_begin = clock();
  for(int n = 0; n < num_; n++)
  {
    for(int h = 0; h < height_; h++)
    {
      for(int w = 0; w < width_; w++)
      {
        // compute the max prob
        Dtype max_value = Dtype(-1);
        int max_id = 0;
        for(int c = 1; c < channels_; c++)
        {
          Dtype value = bottom[0]->data_at(n, c, h, w);
          if(value > max_value)
          {
            max_value = value;
            max_id = c;
          }
        }
        // store the max prob and id
        int index = top[0]->offset(n, 0, h, w);
        heatmap[index] = max_value;
        argmax_data[index] = max_id;
      }
    }
  }
  clock_t time_end = clock();
  double elapsed_secs = double(time_end - time_begin) / CLOCKS_PER_SEC;
  cout << "Compute heatmap: " << elapsed_secs << " second\n";
}

template <typename Dtype>
void HeatmapGeneratingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype* top_diff = top[0]->cpu_diff();

  if (propagate_down[0]) 
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    clock_t time_begin = clock();
    for(int n = 0; n < num_; n++)
    {
      for(int h = 0; h < height_; h++)
      {
        for(int w = 0; w < width_; w++)
        {
          int max_id = max_idx_.data_at(n, 0, h, w);
          int index_bottom = bottom[0]->offset(n, max_id, h, w);
          int index_top = top[0]->offset(n, 0, h, w);
          bottom_diff[index_bottom] = top_diff[index_top];
        }
      }
    }
    clock_t time_end = clock();
    double elapsed_secs = double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "Compute heatmap gradient: " << elapsed_secs << " second\n";
  }
}

INSTANTIATE_CLASS(HeatmapGeneratingLayer);
REGISTER_LAYER_CLASS(HeatmapGenerating);

}  // namespace caffe
