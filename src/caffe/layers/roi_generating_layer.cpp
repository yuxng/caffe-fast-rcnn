// ------------------------------------------------------------------
// Subcategory CNN
// Copyright (c) 2015 CVGL Stanford
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

# include <cfloat>
# include <ctime>
# include <assert.h>
# include <stdio.h>
# include <algorithm>
# include <functional>

#include "caffe/roi_generating_layers.hpp"

namespace caffe {

template <typename Dtype>
void ROIGeneratingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIGeneratingParameter roi_generating_param = this->layer_param_.roi_generating_param();

  CHECK_GT(roi_generating_param.batch_size(), 0)
      << "batch size must be > 0";

  flag_proposal_only_ = roi_generating_param.flag_proposal_only();
  batch_size_ = roi_generating_param.batch_size();
  num_classes_ = roi_generating_param.num_classes();
  fg_fraction_ = roi_generating_param.fg_fraction();
  spatial_scale_ = roi_generating_param.spatial_scale();
}

template <typename Dtype>
void ROIGeneratingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // rois_sub
  top[0]->Reshape(batch_size_, 5, 1, 1);
  // sublabels
  top[1]->Reshape(batch_size_, 1, 1, 1);
  // bbox targets
  top[2]->Reshape(batch_size_, 4 * num_classes_, 1, 1);
  // bbox inside weights
  top[3]->Reshape(batch_size_, 4 * num_classes_, 1, 1);
  // bbox outside weights
  top[4]->Reshape(batch_size_, 4 * num_classes_, 1, 1);

  if(flag_proposal_only_ == 0)
  {
    // rois
    top[5]->Reshape(batch_size_, 5, 1, 1);
    // labels
    top[6]->Reshape(batch_size_, 1, 1, 1);
  }
}

template <typename Dtype>
void ROIGeneratingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_parameters = bottom[2]->cpu_data();
  std::vector<std::pair<Dtype, int> > heatmap(num_ * height_ * width_);

  // parse parameters
  int num_scale = int(bottom_parameters[0]);
  int num_aspect = int(bottom_parameters[1]);
  const Dtype* scales = bottom_parameters + 2;
  const Dtype* scale_mapping = bottom_parameters + 2 + num_scale;
  const Dtype* aspect_heights = bottom_parameters + 2 + 2 * num_scale;
  const Dtype* aspect_widths = bottom_parameters + 2 + 2 * num_scale + num_aspect;

  // numbers
  int num_batch = bottom[0]->num();
  int num_image = num_batch / num_scale;
  int rois_per_image = batch_size_ / num_image;
  int fg_rois_per_image = int(fg_fraction_ * rois_per_image);

  // build the heatmap vector
  // clock_t time_begin = clock();
  for(int i = 0; i < bottom[0]->count(); i++)
    heatmap[i] = std::make_pair(bottom_data[i], i);
  // clock_t time_end = clock();
  // double elapsed_secs = double(time_end - time_begin) / CLOCKS_PER_SEC;
  // cout << "Compute heatmap: " << elapsed_secs << " second\n";

  // process the positive boxes
  int num_positive = bottom[1]->num();
  std::vector<std::pair<Dtype, int> > scores_positive_vector;
  std::vector<int> sep_positive_vector(num_image+1, -1);
  for(int i = 0; i < num_positive; i++)
  {
    int cx = int(bottom[1]->data_at(i, 0, 0, 0));
    int cy = int(bottom[1]->data_at(i, 1, 0, 0));
    int batch_index = int(bottom[1]->data_at(i, 2, 0, 0));
    int index = batch_index * height_ * width_ + cy * width_ + cx;
    scores_positive_vector.push_back(std::make_pair(heatmap[index].first, i));
    // mask the heatmap location
    heatmap[index].first = -1;
    // check which image
    int image_index = batch_index / num_scale;
    sep_positive_vector[image_index+1] = i;
  }

  // select positive boxes for each image
  std::vector<int> index_positive;
  std::vector<int> count_image(num_image, 0);
  for(int i = 0; i < num_image; i++)
  {
    // [start, end)
    int start = sep_positive_vector[i] + 1;
    int end = sep_positive_vector[i+1] + 1;
    int num = end - start;

    if(num <= fg_rois_per_image)
    {
      // use all the positives of this image
      for(int j = start; j < end; j++) 
        index_positive.push_back(j);
      count_image[i] = num;
    }
    else
    {
      // select hard positives (low score positives)
      std::partial_sort(
        scores_positive_vector.begin() + start, scores_positive_vector.begin() + start + fg_rois_per_image,
        scores_positive_vector.begin() + end, std::less<std::pair<Dtype, int> >());

      for(int j = 0; j < fg_rois_per_image; j++) 
        index_positive.push_back(scores_positive_vector[start+j].second);
      count_image[i] = fg_rois_per_image;
    }
  }

  // select negative boxes for each image
  std::vector<int> index_negative;
  for(int i = 0; i < num_image; i++)
  {
    // [start, end)
    int start = i * num_scale * height_ * width_;
    int end = (i+1) * num_scale * height_ * width_;
    int num = rois_per_image - count_image[i];

    // sort heatmap to select hard negatives (high score negatives)
    std::partial_sort(
      heatmap.begin() + start, heatmap.begin() + start + num,
      heatmap.begin() + end, std::greater<std::pair<Dtype, int> >());

    for(int j = 0; j < num; j++)
      index_negative.push_back(heatmap[start+j].second);
  }

  // build the blobs of interest
  Dtype* rois_sub = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), rois_sub);

  Dtype* sublabels = top[1]->mutable_cpu_data();
  caffe_set(top[1]->count(), Dtype(0), sublabels);

  Dtype* bbox_targets = top[2]->mutable_cpu_data();
  caffe_set(top[2]->count(), Dtype(0), bbox_targets);

  Dtype* bbox_inside_weights = top[3]->mutable_cpu_data();
  caffe_set(top[3]->count(), Dtype(0), bbox_inside_weights);

  Dtype* bbox_outside_weights = top[4]->mutable_cpu_data();
  caffe_set(top[4]->count(), Dtype(0), bbox_outside_weights);

  Dtype* rois = NULL;
  Dtype* labels = NULL;
  if(flag_proposal_only_ == 0)
  {
    rois = top[5]->mutable_cpu_data();
    labels = top[6]->mutable_cpu_data();

    caffe_set(top[5]->count(), Dtype(0), rois);
    caffe_set(top[6]->count(), Dtype(0), labels);
  }

  int count = 0;
  // positives
  for(int i = 0; i < index_positive.size(); i++)
  {
    int ind = index_positive[i];

    for(int j = 0; j < 5; j++)
      rois_sub[count*5 + j] = bottom[1]->data_at(ind, 2+j, 0, 0); // info_boxes[ind, 2:7]

    sublabels[count] = bottom[1]->data_at(ind, 13, 0, 0); // info_boxes[ind, 13]

    // bounding box regression
    int cls = int(bottom[1]->data_at(ind, 12, 0, 0));
    int start = 4 * cls;
    for(int j = 0; j < 4; j++)
    {
      bbox_targets[count * 4 * num_classes_ + start + j] = bottom[1]->data_at(ind, 14 + j, 0, 0); // info_boxes[ind, 14:]
      bbox_inside_weights[count * 4 * num_classes_ + start + j] = 1.0;
      bbox_outside_weights[count * 4 * num_classes_ + start + j] = 1.0;
    }

    if(flag_proposal_only_ == 0)
    {
      for(int j = 0; j < 5; j++)
        rois[count*5 + j] = bottom[1]->data_at(ind, 7+j, 0, 0); // info_boxes[ind, 7:12]

      labels[count] = bottom[1]->data_at(ind, 12, 0, 0);    // info_boxes[ind, 12]
    }

    count++;
  }

  /* initialize random seed: */
  srand(time(NULL));

  // negatives
  for(int i = 0; i < index_negative.size(); i++)
  {
    int ind = index_negative[i];

    // parse index
    int batch_index = ind / (height_ * width_);
    int tmp = ind % (height_ * width_);
    int cy = tmp / width_;
    int cx = tmp % width_;

    // sample an aspect ratio
    int aspect_index = rand() % num_aspect;
    Dtype width = aspect_widths[aspect_index];
    Dtype height = aspect_heights[aspect_index];

    // scale mapping
    int image_index = batch_index / num_scale;
    int scale_index = batch_index % num_scale;
    Dtype scale = scales[scale_index];

    // check if the point is inside this scale
    Dtype rescale = scale / scales[num_scale-1];
    Dtype scale_map;
    int batch_index_map;
    if(cx < width_ * rescale && cy < height_ * rescale)
    {
      int scale_index_map = int(scale_mapping[scale_index]);
      scale_map = scales[scale_index_map];
      batch_index_map = image_index * num_scale + scale_index_map;
    }
    else
    {
      // do not do scale mapping
      scale_map = scale;
      batch_index_map = batch_index;
    }

    // assign information
    rois_sub[count * 5 + 0] = batch_index;
    rois_sub[count * 5 + 1] = (cx - width / 2) / spatial_scale_;
    rois_sub[count * 5 + 2] = (cy - height / 2) / spatial_scale_;
    rois_sub[count * 5 + 3] = (cx + width / 2) / spatial_scale_;
    rois_sub[count * 5 + 4] = (cy + height / 2) / spatial_scale_;

    if(flag_proposal_only_ == 0)
    {
      rois[count * 5] = batch_index_map;
      for(int j = 0; j < 4; j++)
        rois[count * 5 + j + 1] = rois_sub[count * 5 + j + 1] * scale_map / scale;
    }

    count++;
  }

  assert(count == batch_size_);
}

template <typename Dtype>
void ROIGeneratingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // Initialize all the gradients to 0
  if (propagate_down[0]) 
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  }
}

INSTANTIATE_CLASS(ROIGeneratingLayer);
REGISTER_LAYER_CLASS(ROIGenerating);

}  // namespace caffe
