#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "darknet.h"
#include "network.h"
#include "option_list.h"
#include "image.h"
#include "data.h"
#include "detector/BBox.h"
#include "detector/DetectResult.h"

#define GPARAM(x, y)                       \
  if (!nh.getParam(x, y)) {                \
    ROS_FATAL("Get pararm " #x " error!"); \
  }

cv::Scalar getColor(int i) {
  return cv::Scalar(((i + 6 << 6) | 123) % 256, ((i + 5) << 5 & 123) % 256,
                    ((i + 4) << 6 | 123) % 256);
}

namespace detect_nodelet {
class DetectNodelet : public nodelet::Nodelet {
 public:
  DetectNodelet() {}
  ~DetectNodelet() {}

 private:
  virtual void onInit() {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle pnh = getPrivateNodeHandle();

    image_transport::ImageTransport it(nh);
    // Result publisher
    res_pub_ = pnh.advertise<detector::DetectResult>("detect_result", 1);

    // Get params
    GPARAM("/detect/num_gpus", num_gpus_);
    GPARAM("/detect/num_types", num_types_);
    show_gui_ = nh.param<bool>("show_gui", false);

    weights_.resize(num_types_);
    datas_.resize(num_types_);
    image_.resize(num_types_);
    // Start thread
    for (int i = 0; i < num_types_; i++) {
      // Image subscriber
      image_sub_.emplace_back(it.subscribe(
          "cam_" + std::to_string(i) + "/image", 1,
          boost::bind(&DetectNodelet::imageCallback, this, _1, i)));
      // Two inference thread, load weights and wait for flag to do detction
      detect_t_.push_back(std::thread([this, i, nh] {
        std::string prefix = "detect/cam_" + std::to_string(i);
        std::string test_cfg;

        // Read network cfgs
        GPARAM(prefix + "/data_file", datas_[i]);
        GPARAM(prefix + "/test_weights", weights_[i]);
        GPARAM(prefix + "/test_cfg", test_cfg);
        char *data_file = (char *)malloc(datas_[i].length());
        char *cfg_file = (char *)malloc(test_cfg.length());
        char *weight_file = (char *)malloc(weights_[i].length());
        strcpy(data_file, datas_[i].c_str());
        strcpy(cfg_file, test_cfg.c_str());
        strcpy(weight_file, weights_[i].c_str());
        testDetector(data_file, cfg_file, weight_file, 0.5, 0.5, 0, 0, i);
      }));
    }
    if (show_gui_) {
      for (int i = 0; i < num_types_; i++)
        cv::namedWindow("cam_" + std::to_string(i), 0);
      cv::startWindowThread();
    }
  }

  int num_gpus_;
  int num_types_;
  std::vector<std::string> weights_;
  std::vector<std::string> datas_;

  std::vector<int> detect_flag_ = {0};

  std::vector<std::thread> detect_t_;

  std::string img_path_;

  void testDetector(char *datacfg, char *cfgfile, char *weightfile,
                    float thresh, float hier_thresh, char *outfile,
                    int fullscreen, int cam_type);

  void imageCallback(const sensor_msgs::ImageConstPtr &msg, int cam_id);

  std::vector<image_transport::Subscriber> image_sub_;
  std::vector<cv::Mat> image_;
  std::mutex image_lock_[5];

  ros::Publisher res_pub_;
  bool show_gui_ = false;
};

void DetectNodelet::imageCallback(const sensor_msgs::ImageConstPtr &msg,
                                  int cam_id) {
  int cam_type;
  if (cam_id > 60)
    cam_type = num_types_;
  else
    cam_type = cam_id;
  {
    std::lock_guard<std::mutex> lock(image_lock_[cam_type]);
    if (!image_[cam_type].empty()) image_[cam_type].release();
    image_[cam_type] = cv_bridge::toCvShare(msg, "bgr8")->image;
  }
  if (cam_type == num_types_)
    detect_flag_[num_types_] = cam_id;
  else
    detect_flag_[cam_type] = 1;
}

void DetectNodelet::testDetector(char *datacfg, char *cfgfile, char *weightfile,
                                 float thresh, float hier_thresh, char *outfile,
                                 int fullscreen, int cam_type) {
  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  // image **alphabet = load_alphabet();
  printf("Loading network config...\n");
  network *net = load_network(cfgfile, weightfile, 0);
  set_batch_network(net, 1);
  srand(2222222);
  double time;
  char buff[256];
  char *input = buff;
  float nms = .45;
  while (1) {
    if (detect_flag_[cam_type]) {
      image im;
      {
        std::lock_guard<std::mutex> lock(image_lock_[cam_type]);
        if (image_[cam_type].empty()) {
          detect_flag_[cam_type] = false;
          continue;
        }
        // Convert Mat to image
        int h = image_[cam_type].rows;
        int w = image_[cam_type].cols;
        int c = image_[cam_type].channels();
        im = make_image(w, h, c);
        unsigned char *data = (unsigned char *)image_[cam_type].data;
        int step = image_[cam_type].step;
        int i, j, k;
        for (i = 0; i < h; ++i) {
          for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
              im.data[k * w * h + i * w + j] =
                  data[i * step + j * c + k] / 255.;
            }
          }
        }

        // im = mat_to_image(image_[cam_type]);
      }

      image sized = letterbox_image(im, net->w, net->h);
      layer l = net->layers[net->n - 1];

      float *X = sized.data;
      time = what_time_is_it_now();
      network_predict(*net, X);
      ROS_INFO("%s: Predicted in %f seconds.\n", input,
             what_time_is_it_now() - time);
      int nboxes = 0;
      detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh,
                                          0, 1, &nboxes,1);

      if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

      detector::DetectResult result_msg;
      if (cam_type == num_types_)
        result_msg.cam_type = detect_flag_[cam_type];
      else
        result_msg.cam_type = cam_type;
      result_msg.bboxes.clear();
      result_msg.num_result = nboxes;
      for (int i = 0; i < nboxes; i++) {
        detector::BBox bbox;
        double max_prob = -1;
        box yolo_box;
        for (int j = 0; j < l.classes; ++j) {
          if (dets[i].prob[j] > thresh) {
            if (dets[i].prob[j] > max_prob) {
              bbox.label = j;
              max_prob = dets[i].prob[j];
              yolo_box = dets[i].bbox;
            }
          }
        }
        // Ignore invalid box
        if (max_prob < 0) {
          result_msg.num_result--;
          continue;
        }
        bbox.x_min = (yolo_box.x - yolo_box.w / 2.) * im.w;
        bbox.x_max = (yolo_box.x + yolo_box.w / 2.) * im.w;
        bbox.y_min = (yolo_box.y - yolo_box.h / 2.) * im.h;
        bbox.y_max = (yolo_box.y + yolo_box.h / 2.) * im.h;
        if (bbox.x_min < 0) bbox.x_min = 0;
        if (bbox.x_max > im.w - 1) bbox.x_max = im.w - 1;
        if (bbox.y_min < 0) bbox.y_min = 0;
        if (bbox.y_max > im.h - 1) bbox.y_max = im.h - 1;
        bbox.prob = max_prob;
        result_msg.bboxes.push_back(bbox);
      }
      free_detections(dets, nboxes);

      free_image(im);
      free_image(sized);
      res_pub_.publish(result_msg);
      detect_flag_[cam_type] = 0;

      if (show_gui_) {
        for (auto box : result_msg.bboxes)
          cv::rectangle(image_[cam_type], cv::Point(box.x_min, box.y_min),
                        cv::Point(box.x_max, box.y_max), getColor(box.label),
                        5);
        cv::imshow("cam_" + std::to_string(result_msg.cam_type),
                   image_[cam_type]);
      }
    } else
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace detect_nodelet

PLUGINLIB_EXPORT_CLASS(detect_nodelet::DetectNodelet, nodelet::Nodelet);