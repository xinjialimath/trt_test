#include "convert_model.hpp"

void ConvertModel::onInit(){
  std::ifstream fs(engine_path);
  if (readLabelFile(filepath, labels)) {
    if (!getTlrIdFromLabel(labels, tlr_id_)) {
      std::cout << "Could not find tlr id" << std::endl;
    }
  } //tlt_id = 1; labels = [BACKGROUND, traffic_light]

  if (fs.is_open()) { // if there is an engine, check if max_batch_size matches.
    std::cout << "Found " <<  engine_path.c_str() << std::endl;
    net_ptr_.reset(new ssd::Net(engine_path, false));
    if (max_batch_size != net_ptr_->getMaxBatchSize()) {
    std::cout << "Required max batch size does not correspond to Profile max batch size. Rebuild " << std::endl;
      net_ptr_.reset(new ssd::Net(onnx_file, mode, max_batch_size));
      net_ptr_->save(engine_path);
    }
  } else { //if there is no engine, just generate it.
    std::cout << "Could not find " <<  engine_path.c_str() << " try making TensorRT engine from onnx" << std::endl;
    net_ptr_.reset(new ssd::Net(onnx_file, mode, max_batch_size));
    net_ptr_->save(engine_path);
  }

  channel_ = net_ptr_->getInputSize()[0];
  width_ = net_ptr_->getInputSize()[1];
  height_ = net_ptr_->getInputSize()[2];
  detection_per_class_ = net_ptr_->getOutputScoreSize()[0];
  class_num_ = net_ptr_->getOutputScoreSize()[1];

  cv::Mat img = cv::imread(image_file);
  in_imgs.push_back(img);
}

bool ConvertModel::getTlrIdFromLabel( //tlr_id=1
  const std::vector<std::string> & labels, int & tlr_id)
{
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels.at(i) == "traffic_light") {
      tlr_id = i;
      return true;
    }
  }
  return false;
}

void ConvertModel::infer(){
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  const auto exe_start_time = high_resolution_clock::now();
  int num_rois = 1; //TODO:check if num_rois is ok
  int batch_count = 0;
  const int batch_size = net_ptr_->getMaxBatchSize(); //8
  while (num_rois != 0) {
      const int num_infer = 1; //(num_rois / batch_size > 0) ? batch_size : num_rois % batch_size;//
      auto data_d = cuda::make_unique<float[]>(num_infer * channel_ * width_ * height_); //input tensor
      auto scores_d = cuda::make_unique<float[]>(num_infer * detection_per_class_ * class_num_); //out
      auto boxes_d = cuda::make_unique<float[]>(num_infer * detection_per_class_ * 4); //bboxes vector
      std::vector<void *> buffers = {data_d.get(), scores_d.get(), boxes_d.get()};
      std::vector<cv::Point> lts, rbs;
    //  std::vector<cv::Mat> cropped_imgs;
      std::vector<float> data(num_infer * channel_ * width_ * height_);
      if (!cvMat2CnnInput(in_imgs, num_infer, data)) { //对input进行预处理
         std::cout << "Fail to preprocess image" << std::endl;
         return;
      }
      //copy to gpu
      cudaMemcpy(data_d.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);

      try {
         net_ptr_->infer(buffers, num_infer); //infer
      } catch (std::exception & e) {
          std::cout << e.what() << std::endl;
          return;
        }
      //allocate the scores and boxes space
      auto scores = std::make_unique<float[]>(num_infer * detection_per_class_ * class_num_);
      auto boxes = std::make_unique<float[]>(num_infer * detection_per_class_ * 4);
      //copy to cpu
      cudaMemcpy(
        scores.get(), scores_d.get(), sizeof(float) * num_infer * detection_per_class_ * class_num_,
           cudaMemcpyDeviceToHost);
      cudaMemcpy(
        boxes.get(), boxes_d.get(), sizeof(float) * num_infer * detection_per_class_ * 4,
          cudaMemcpyDeviceToHost);
      // Get Output
      std::vector<Detection> detections;

      //Post process
      if (!cnnOutput2BoxDetection(
          scores.get(), boxes.get(), tlr_id_, in_imgs, num_infer, detections)) {
          std::cout << "Fail to postprocess image" << std::endl;
          return;
        }

      // for (int i = 0; i < num_infer; ++i) {
      //   if (detections.at(i).prob > score_thresh_) {
      //     cv::Point lt_roi =
      //       cv::Point(lts.at(i).x + detections.at(i).x, lts.at(i).y + detections.at(i).y);
      //     cv::Point rb_roi = cv::Point(
      //       lts.at(i).x + detections.at(i).x + detections.at(i).w,
      //       lts.at(i).y + detections.at(i).y + detections.at(i).h);
      //     fitInFrame(lt_roi, rb_roi, cv::Size(in_imgs[0].size()));
      //       out_rois.rois.push_back(tl_roi);
      //     }
      //   }
        num_rois -= num_infer;
        ++batch_count;
  }
  const auto exe_end_time = high_resolution_clock::now();
  const double exe_time =
    std::chrono::duration_cast<milliseconds>(exe_end_time - exe_start_time).count();
  std::cout << "the exe_time is: " << exe_time << std::endl;
}

bool ConvertModel::fitInFrame(cv::Point & lt, cv::Point & rb, const cv::Size & size)
{ //对超过边界的进行处理，处理为边界
  try {
    if (rb.x > size.width) rb.x = size.width;
    if (rb.y > size.height) rb.y = size.height;
    if (lt.x < 0) lt.x = 0;
    if (lt.y < 0) lt.y = 0;
  } catch (cv::Exception & e) {
    std::cout << "Failed to fit bounding rect in size" << std::endl;
    return false;
  }
}

bool ConvertModel::cnnOutput2BoxDetection( //post precessing
  const float * scores, const float * boxes, const int tlr_id, const std::vector<cv::Mat> & in_imgs,
  const int num_rois, std::vector<Detection> & detections)
{
  if (tlr_id > class_num_ - 1) {
    return false;
  }
  for (int i = 0; i < num_rois; ++i) {
    std::vector<float> tlr_scores;
    Detection det;
    for (int j = 0; j < detection_per_class_; ++j) {
      tlr_scores.push_back(scores[i * detection_per_class_ * class_num_ + tlr_id + j * class_num_]);
    }
    std::vector<float>::iterator iter = std::max_element(tlr_scores.begin(), tlr_scores.end());
    size_t index = std::distance(tlr_scores.begin(), iter);
    size_t box_index = i * detection_per_class_ * 4 + index * 4;
    det.x = boxes[box_index] * in_imgs.at(i).cols;
    det.y = boxes[box_index + 1] * in_imgs.at(i).rows;
    det.w = (boxes[box_index + 2] - boxes[box_index]) * in_imgs.at(i).cols;
    det.h = (boxes[box_index + 3] - boxes[box_index + 1]) * in_imgs.at(i).rows;
    det.prob = tlr_scores[index];
    detections.push_back(det);
  }
  return true;
}

bool ConvertModel::cvMat2CnnInput( //对输入图像的预处理
  const std::vector<cv::Mat> & in_imgs, const int num_rois, std::vector<float> & data)
{
  for (int i = 0; i < num_rois; ++i) {
    // cv::Mat rgb;
    // cv::cvtColor(in_imgs.at(i), rgb, CV_BGR2RGB);
    cv::Mat resized; //将step 1给出的rois resize为cnn输入size
    cv::resize(in_imgs.at(i), resized, cv::Size(width_, height_));

    cv::Mat pixels;
    resized.convertTo(pixels, CV_32FC3, 1.0 / 255, 0);
    std::vector<float> img;
    if (pixels.isContinuous()) {
      img.assign((float *)pixels.datastart, (float *)pixels.dataend);
    } else {
      return false;
    }

    for (int c = 0; c < channel_; ++c) {
      for (int j = 0, hw = width_ * height_; j < hw; ++j) {
        data[i * channel_ * width_ * height_ + c * hw + j] =
          (img[channel_ * j + 2 - c] - mean_[c]) / std_[c];
      }
    }
  }
  return true;
}

bool ConvertModel::readLabelFile(std::string filepath, std::vector<std::string> & labels){
  std::ifstream labelsFile(filepath);
  if (!labelsFile.is_open()) {
    std::cout << "Could not open label file: " <<  filepath.c_str() << std::endl;
    return false;
  }
  std::string label;
  while (getline(labelsFile, label)) {
    labels.push_back(label);
  }
  return true;
} //labels = [BACKGROUND, traffic_light]

