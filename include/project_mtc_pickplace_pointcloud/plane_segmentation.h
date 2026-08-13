#ifndef PLANE_SEGMENTATION_H
#define PLANE_SEGMENTATION_H

#include <iostream>
#include <memory>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <tuple>

#define LOG_INFO(x) std::cout << "INFO: " << x << std::endl
#define LOG_ERROR(x) std::cout << "ERROR: " << x << std::endl

std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::ModelCoefficients::Ptr>
segmentPlaneAndObjects(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
    bool enable_cropping = false,
    double crop_min_x = -std::numeric_limits<double>::max(),
    double crop_max_x = std::numeric_limits<double>::max(),
    double crop_min_y = -std::numeric_limits<double>::max(),
    double crop_max_y = std::numeric_limits<double>::max(),
    double crop_min_z = -std::numeric_limits<double>::max(),
    double crop_max_z = std::numeric_limits<double>::max(),
    int max_iteration = 100, double distance_threshold = 0.01,
    double z_tolerance = 0.03, double angle_tolerance = cos(2.5 * M_PI / 180.0),
    int min_cluster_size = 100, int max_cluster_size = 25000,
    double cluster_tolerance = 0.02, int normal_estimation_k = 30,
    double plane_segmentation_threshold = 0.001, double w_inliners = 1.0,
    double w_size = 1.0, double w_distance = 1.0, double w_orientation = 1.0);

#endif