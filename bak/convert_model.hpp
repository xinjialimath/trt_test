#pragma once

#include <string>
#include <chrono>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "trt_ssd.h"
#include "cuda_utils.h"

typedef struct Detection
{
    float x, y, w, h, prob;
}Detection;

class ConvertModel
{
    public:
    void onInit();
    bool readLabelFile(std::string filepath, std::vector<std::string> & labels);
    bool getTlrIdFromLabel(const std::vector<std::string> & labels, int & tlr_id);
    void infer();
    bool fitInFrame(cv::Point & lt, cv::Point & rb, const cv::Size & size);
    bool cnnOutput2BoxDetection(const float * scores, const float * boxes, const int tlr_id, const std::vector<cv::Mat> & in_imgs,
            const int num_rois, std::vector<Detection> & detections);
    bool cvMat2CnnInput(const std::vector<cv::Mat> & in_imgs, const int num_rois, std::vector<float> & data);

    std::string image_file = "data/test.jpg";
    std::string data_path =  "data/";
    std::string engine_path =  "data/mb2-ssd-lite.engine";
    std::string onnx_file = "data/mb2-ssd-lite-tlr.onnx";
    std::string filepath = "data/voc_labels_tl2.txt";
    std::string mode = "FP32";
    std::vector<std::string> labels;
    std::vector<cv::Mat> in_imgs;
    int max_batch_size = 1;
    double score_thresh_ = 0.7;

    int tlr_id_;
    int channel_;
    int width_;
    int height_;
    int class_num_;
    int detection_per_class_;

    std::vector<float> mean_;
    std::vector<float> std_;

    std::unique_ptr<ssd::Net> net_ptr_;

};