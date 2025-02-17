//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include <cstdlib>
#include <string>
#include <memory>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <tuple>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <ctime>
#include <iomanip>

#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <NvInfer.h>

#include "3rdparty/tensorrtbuffer/include/logger.h"
#include "3rdparty/tensorrtbuffer/include/buffers.h"

#include "./superpoint_mono_trt.h"
#include "./lightglue_trt.h"


using namespace tensorrt_log;
using namespace tensorrt_buffer;

std::vector<std::string> GetFileNames(const std::string& path);

void VisualizeMatching(
    const cv::Mat& image_0, const std::vector<cv::KeyPoint>& keypoints_0, 
    const cv::Mat& image_1, const std::vector<cv::KeyPoint>& keypoints_1, 
    const std::vector<cv::DMatch>& matches, 
    cv::Mat& output_image
);

int main(int argc, char** argv) {

    if (argc != 7) {
        std::cerr << "Six cli parameters." << std::endl;
        return 0;
    }

    const std::string superpoint_mono_engine_path = argv[1];
    const std::string lightglue_engine_path       = argv[2];
    const std::string lightglue_plugin_path       = argv[3];
    const std::string superpoint_image_0_path     = argv[4];
    const std::string superpoint_image_1_path     = argv[5];
    const std::string matches_result_dir          = argv[6];

    // --------------------------------------------------------------------------------

    SuperPointMonoTRT superpoint_mono(superpoint_mono_engine_path, gLogger);
    LightGlueTRT lightglue(lightglue_engine_path, lightglue_plugin_path, gLogger);

    // --------------------------------------------------------------------------------

    std::unordered_map<std::string, std::vector<int64_t>> superpoint_mono_input_shape;
    std::unordered_map<std::string, torch::Tensor> superpoint_mono_input_tensor;
    std::unordered_map<std::string, std::vector<int64_t>> lightglue_input_shape;
    std::unordered_map<std::string, torch::Tensor> lightglue_input_tensor;

    // --------------------------------------------------------------------------------

    // open image_0

    cv::Mat image_0_temp = cv::imread(superpoint_image_0_path, cv::IMREAD_GRAYSCALE);
    if (image_0_temp.empty()) {
        std::cerr << "Input image is empty. Please check the image path." << std::endl;
        return 0;
    }
    cv::resize(image_0_temp, image_0_temp, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
    cv::Mat image_0_cv; image_0_temp.convertTo(image_0_cv, CV_32F); image_0_cv /= 255.0f;
    auto image_0_fp32_cpu = torch::zeros(
        {1, 1, image_0_cv.rows, image_0_cv.cols}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)
    );
    auto image_0_fp32_cpu_accessor = image_0_fp32_cpu.accessor<float, 4>(); // using image_0_fp32_cpu_accessor is faster
    for (int r = 0; r < image_0_cv.rows; r++) {
        for (int c = 0; c < image_0_cv.cols; c++) {
            image_0_fp32_cpu_accessor[0][0][r][c] = image_0_cv.at<float>(r, c);
        }
    }

    // --------------------------------------------------------------------------------

    superpoint_mono_input_shape.clear();
    superpoint_mono_input_tensor.clear();
    superpoint_mono_input_shape["image"] = {1, 1, image_0_fp32_cpu.size(2), image_0_fp32_cpu.size(3)};
    superpoint_mono.SetInputShape(superpoint_mono_input_shape);
    superpoint_mono_input_tensor["image"] = image_0_fp32_cpu;
    superpoint_mono.CopyInputTensor(superpoint_mono_input_tensor);
    auto superpoint_output_tuple_0 = superpoint_mono.Forward();
    auto superpoint_keypoints_0_int32_cuda           = std::get<0>(superpoint_output_tuple_0);
    auto superpoint_keypoints_normalized_0_fp32_cuda = std::get<1>(superpoint_output_tuple_0).unsqueeze(0);
    auto superpoint_descriptors_0_fp32_cuda          = std::get<2>(superpoint_output_tuple_0).unsqueeze(0);

    // --------------------------------------------------------------------------------

    // open image_1

    cv::Mat image_1_temp = cv::imread(superpoint_image_1_path, cv::IMREAD_GRAYSCALE);
    if (image_1_temp.empty()) {
        std::cerr << "Input image is empty. Please check the image path." << std::endl;
        return 0;
    }
    cv::resize(image_1_temp, image_1_temp, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
    cv::Mat image_1_cv; image_1_temp.convertTo(image_1_cv, CV_32F); image_1_cv /= 255.0f;
    auto image_1_fp32_cpu = torch::zeros(
        {1, 1, image_1_cv.rows, image_1_cv.cols}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)
    );
    auto image_1_fp32_cpu_accessor = image_1_fp32_cpu.accessor<float, 4>();
    for (int r = 0; r < image_1_cv.rows; r++) {
        for (int c = 0; c < image_1_cv.cols; c++) {
            image_1_fp32_cpu_accessor[0][0][r][c] = image_1_cv.at<float>(r, c);
        }
    }

    // --------------------------------------------------------------------------------

    superpoint_mono_input_shape.clear();
    superpoint_mono_input_tensor.clear();
    superpoint_mono_input_shape["image"] = {1, 1, image_1_fp32_cpu.size(2), image_1_fp32_cpu.size(3)};
    superpoint_mono.SetInputShape(superpoint_mono_input_shape);
    superpoint_mono_input_tensor["image"] = image_1_fp32_cpu;
    superpoint_mono.CopyInputTensor(superpoint_mono_input_tensor);
    auto superpoint_output_tuple_1 = superpoint_mono.Forward();
    auto superpoint_keypoints_1_int32_cuda           = std::get<0>(superpoint_output_tuple_1);
    auto superpoint_keypoints_normalized_1_fp32_cuda = std::get<1>(superpoint_output_tuple_1).unsqueeze(0);
    auto superpoint_descriptors_1_fp32_cuda          = std::get<2>(superpoint_output_tuple_1).unsqueeze(0);

    // --------------------------------------------------------------------------------

    lightglue_input_shape.clear();
    lightglue_input_tensor.clear();
    lightglue_input_shape["keypoints_0"] = {1, superpoint_keypoints_normalized_0_fp32_cuda.size(1), 2};
    lightglue_input_shape["keypoints_1"] = {1, superpoint_keypoints_normalized_1_fp32_cuda.size(1), 2};
    lightglue_input_shape["descriptors_0"] = {1, superpoint_descriptors_0_fp32_cuda.size(1), 256};
    lightglue_input_shape["descriptors_1"] = {1, superpoint_descriptors_1_fp32_cuda.size(1), 256};
    lightglue.SetInputShape(lightglue_input_shape);
    lightglue_input_tensor["keypoints_0"] = superpoint_keypoints_normalized_0_fp32_cuda;
    lightglue_input_tensor["keypoints_1"] = superpoint_keypoints_normalized_1_fp32_cuda;
    lightglue_input_tensor["descriptors_0"] = superpoint_descriptors_0_fp32_cuda;
    lightglue_input_tensor["descriptors_1"] = superpoint_descriptors_1_fp32_cuda;
    lightglue.CopyInputTensor(lightglue_input_tensor);
    auto lightglue_output_tuple = lightglue.Forward();
    auto lightglue_descriptors_0_fp32_cuda    = std::get<0>(lightglue_output_tuple);
    auto lightglue_descriptors_1_fp32_cuda    = std::get<1>(lightglue_output_tuple);
    auto lightglue_scores_fp32_cuda           = std::get<2>(lightglue_output_tuple);
    auto lightglue_match_indices_0_int64_cuda = std::get<3>(lightglue_output_tuple);
    auto lightglue_match_indices_1_int64_cuda = std::get<4>(lightglue_output_tuple);
    auto lightglue_match_scores_fp32_cuda     = std::get<5>(lightglue_output_tuple);

    // --------------------------------------------------------------------------------

    auto match_keypoints_0_int32_cuda = torch::index_select(superpoint_keypoints_0_int32_cuda, 0, lightglue_match_indices_0_int64_cuda);    // [col, row] = [640, 480], same as opencv
    auto match_keypoints_1_int32_cuda = torch::index_select(superpoint_keypoints_1_int32_cuda, 0, lightglue_match_indices_1_int64_cuda);    // [col, row] = [640, 480], same as opencv
    std::cout << "lightglue_match_indices_0_int64_cuda : " << lightglue_match_indices_0_int64_cuda.sizes() << std::endl;
    std::cout << "lightglue_match_indices_1_int64_cuda : " << lightglue_match_indices_1_int64_cuda.sizes() << std::endl;

    assert(match_keypoints_0_int32_cuda.size(0) == match_keypoints_1_int32_cuda.size(0));

    auto match_keypoints_0_int32_cpu = match_keypoints_0_int32_cuda.to(torch::kCPU).to(torch::kInt32);
    auto match_keypoints_1_int32_cpu = match_keypoints_1_int32_cuda.to(torch::kCPU).to(torch::kInt32);
    auto match_keypoints_0_int32_cpu_accessor = match_keypoints_0_int32_cpu.accessor<int, 2>();
    auto match_keypoints_1_int32_cpu_accessor = match_keypoints_1_int32_cpu.accessor<int, 2>();

    std::vector<cv::Point2f> keypoints_0_ransac_vec;
    std::vector<cv::Point2f> keypoints_1_ransac_vec;
    for (int i = 0; i < match_keypoints_0_int32_cpu.size(0); i++) {
        float x;
        float y;
        x = float(match_keypoints_0_int32_cpu_accessor[i][0]);
        y = float(match_keypoints_0_int32_cpu_accessor[i][1]);
        keypoints_0_ransac_vec.emplace_back(x, y);
        x = float(match_keypoints_1_int32_cpu_accessor[i][0]);
        y = float(match_keypoints_1_int32_cpu_accessor[i][1]);
        keypoints_1_ransac_vec.emplace_back(x, y);
    }
    std::vector<uchar> inliers_vec;
    cv::findFundamentalMat(
        keypoints_0_ransac_vec, 
        keypoints_1_ransac_vec, 
        cv::FM_RANSAC, 
        3, 
        0.99, 
        inliers_vec
    );

    std::cout << "inliers_vec.size() : " << inliers_vec.size() << std::endl;

    std::vector<cv::KeyPoint> keypoints_0_vec;
    std::vector<cv::KeyPoint> keypoints_1_vec;
    std::vector<cv::DMatch> matches_vec;
    int maches_cnt = 0;
    for (int i = 0; i < match_keypoints_0_int32_cpu.size(0); i++) {
        float x;
        float y;
        if (inliers_vec[i]) {
            x = float(match_keypoints_0_int32_cpu_accessor[i][0]);
            y = float(match_keypoints_0_int32_cpu_accessor[i][1]);
            keypoints_0_vec.emplace_back(x, y, 8);
            x = float(match_keypoints_1_int32_cpu_accessor[i][0]);
            y = float(match_keypoints_1_int32_cpu_accessor[i][1]);
            keypoints_1_vec.emplace_back(x, y, 8);
            matches_vec.emplace_back(maches_cnt, maches_cnt, 0.0f);
            maches_cnt++;
        }
    }

    std::cout << "matches_vec.size() : " << matches_vec.size() << std::endl;

    for (int start = 0; start < matches_vec.size(); start += 50) {
        cv::Mat temp_matches_image;
        std::vector<cv::DMatch> temp_matches_vec;

        temp_matches_vec.clear();
        int end = (start + 50 - 1) <= (matches_vec.size() - 1) ? (start + 50 - 1) : (matches_vec.size() - 1);
        for (int j = start; j <= end; j++) {
            temp_matches_vec.push_back(matches_vec[j]);
        }

        temp_matches_image.release();
        VisualizeMatching(
            image_0_temp, keypoints_0_vec, 
            image_1_temp, keypoints_1_vec, 
            temp_matches_vec, 
            temp_matches_image
        );

        std::string temp_image_path = matches_result_dir + std::string("/matches_image_") + std::to_string(start) + std::string("_to_") + std::to_string(end) + std::string(".png");
        cv::imwrite(temp_image_path, temp_matches_image);
    }

    // --------------------------------------------------------------------------------

    return 0;
}


void VisualizeMatching(
    const cv::Mat& image_0, const std::vector<cv::KeyPoint>& keypoints_0, 
    const cv::Mat& image_1, const std::vector<cv::KeyPoint>& keypoints_1, 
    const std::vector<cv::DMatch>& matches, 
    cv::Mat& output_image
) {
    if (image_0.size != image_1.size) {
        return;
    }

    cv::drawMatches(
        image_0, keypoints_0, 
        image_1, keypoints_1, 
        matches, 
        output_image, 
        cv::Scalar(0, 255, 0), 
        cv::Scalar(0, 0, 255)
    );
}


std::vector<std::string> GetFileNames(const std::string& path) {
    DIR *pDir;
    struct dirent *ptr;
    std::vector<std::string> filenames;

    if (!(pDir = opendir(path.c_str()))) {
        std::cerr << "Current folder doesn't exist!" << std::endl;
        return std::vector<std::string>();
    }
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(filenames.begin(), filenames.end());

    return filenames;
}

    // auto superpoint_keypoints_normalized_0_fp32_cuda = (torch::rand({1, 1024, 2}, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)) * 2.0f - 1.0f).contiguous();
    // auto superpoint_keypoints_normalized_1_fp32_cuda = (torch::rand({1, 1024, 2}, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)) * 2.0f - 1.0f).contiguous();
    // auto superpoint_descriptors_0_fp32_cuda = (torch::rand({1, 1024, 256}, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)) * 2.0f - 1.0f).contiguous();
    // auto superpoint_descriptors_1_fp32_cuda = (torch::rand({1, 1024, 256}, torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCPU).requires_grad(false)) * 2.0f - 1.0f).contiguous();


