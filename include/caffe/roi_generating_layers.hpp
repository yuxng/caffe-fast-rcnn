// ------------------------------------------------------------------
// Subcategory-CNN
// Copyright (c) 2015 CVGL Stanford
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

#ifndef CAFFE_ROI_GENERATING_LAYERS_HPP_
#define CAFFE_ROI_GENERATING_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* ROIGeneratingLayer - Region of Interest Generating Layer
*/
template <typename Dtype>
class ROIGeneratingLayer : public Layer<Dtype> {
 public:
  explicit ROIGeneratingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIGenerating"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline int MaxTopBlobs() const { return 6; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int height_;
  int width_;

  int flag_proposal_only_;
  int batch_size_;
  int num_classes_;
  float fg_fraction_;
  float spatial_scale_;  
};

/* HeatmapGeneratingLayer */
template <typename Dtype>
class HeatmapGeneratingLayer : public Layer<Dtype> {
 public:
  explicit HeatmapGeneratingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HeatmapGenerating"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_;
  int channels_;
  int height_;
  int width_;

  Blob<int> max_idx_;
};


/* FeatureExtrapolatingLayer */
template <typename Dtype>
class FeatureExtrapolatingLayer : public Layer<Dtype> {
 public:
  explicit FeatureExtrapolatingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "FeatureExtrapolating"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_top_;
  int num_;
  int channels_;
  int height_;
  int width_;

  int channels_trace_;
  int num_image_;
  int num_scale_;
  int num_scale_base_;
  int num_per_octave_;
  float min_scale_;
  float max_scale_;

  Blob<int> is_real_scales_;
  Blob<int> which_base_scales_;
  Blob<double> rescaling_factors_;
  Blob<double> trace_;
};


}  // namespace caffe

#endif  // CAFFE_ROI_GENERATING_LAYERS_HPP_
