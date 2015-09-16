// ------------------------------------------------------------------
// Subcategory CNN
// Copyright (c) 2015 Stanford CVGL
// Licensed under The MIT License
// Written by Yu Xiang
// ------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/roi_generating_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

typedef ::testing::Types<DoubleGPU> TestDtypesGPU;

template <typename TypeParam>
class FeatureExtrapolatingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  FeatureExtrapolatingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(5, 1, 12, 8)),
        blob_top_data_(new Blob<Dtype>()) 
  {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~FeatureExtrapolatingLayerTest() 
  {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(FeatureExtrapolatingLayerTest, TestDtypesGPU);

TYPED_TEST(FeatureExtrapolatingLayerTest, TestGradient) 
{
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  FeatureExtrapolatingParameter* feature_extrapolating_param =
      layer_param.mutable_feature_extrapolating_param();
  feature_extrapolating_param->set_num_scale_base(5);
  feature_extrapolating_param->set_scale_string("0.25 0.5 1.0 2.0 3.0");
  feature_extrapolating_param->set_num_per_octave(4);
  FeatureExtrapolatingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
