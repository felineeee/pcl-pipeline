#ifndef OBJECT_SEGMENTATION_H
#define OBJECT_SEGMENTATION_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pcl/common/io.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include "project_mtc_pickplace_pointcloud/normals_curvature_and_rsd_estimation.h"

Eigen::Vector3f fitLine(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2);
float distanceToLine(const Eigen::Vector2f &point, const Eigen::Vector3f &line);
std::tuple<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr>
fitLineRANSAC(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud,
              double ransac_distance_threshold, int ransac_max_iterations,
              double max_allowable_radius,
              const std::unordered_map<size_t, size_t> &projection_map);
Eigen::Vector3f fitCircle(const Eigen::Vector2f &p1, const Eigen::Vector2f &p2,
                          const Eigen::Vector2f &p3);
float distanceToCircle(const Eigen::Vector2f &point,
                       const Eigen::Vector3f &circle);
std::tuple<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr>
fitCircleRANSAC(const pcl::PointCloud<pcl::PointXY>::Ptr &cloud,
                double ransac_distance_threshold, int ransac_max_iterations,
                double max_allowable_radius,
                const std::unordered_map<size_t, size_t> &projection_map);
void logModelResults(const std::string &model_type,
                     const pcl::ModelCoefficients::Ptr &coefficients,
                     const pcl::PointIndices::Ptr &inliers);
pcl::PointIndices::Ptr filterCircleInliers(
    const pcl::PointIndices::Ptr &circle_inliers,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
        &original_cloud, // PointXYZRGBNormalRSD
    const std::unordered_map<size_t, size_t> &projection_map,
    int circle_min_cluster_size, int circle_max_cluster_size,
    double circle_height_tolerance, double circle_curvature_threshold,
    double circle_radius_tolerance, double circle_normal_angle_threshold,
    double circle_cluster_tolerance);
pcl::PointIndices::Ptr filterLineInliers(
    const pcl::PointIndices::Ptr &line_inliers,
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &original_cloud,
    const std::unordered_map<size_t, size_t> &projection_map,
    int line_min_cluster_size, int line_max_cluster_size,
    double line_curvature_threshold, double line_cluster_tolerance);
struct ValidModel {
  std::string type;
  std::vector<double> parameters;
  std::vector<int> inlier_indices;
};
struct HoughBin {
  std::vector<int> indices;
  int votes;
  int inlierCount;
  std::vector<double> parameters;
};
struct LineModel {
  double rho;
  double theta;
  int votes;
  int inlierCount;
};
std::vector<HoughBin>
clusterLineModels(const std::vector<LineModel> &line_models,
                  double rhoThreshold, double thetaThreshold);
std::vector<HoughBin>
clusterCircleModels(const Eigen::Tensor<int, 3> &houghSpaceCircle,
                    double houghCenterXStep, double houghCenterYStep,
                    double houghRadiusStep, double minPt_x, double minPt_y);
std::tuple<shape_msgs::msg::SolidPrimitive, geometry_msgs::msg::Pose>
fitCylinderToCluster(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cluster,
    double center_x, double center_y, double radius);

std::tuple<shape_msgs::msg::SolidPrimitive, geometry_msgs::msg::Pose>
fitBoxToCluster(const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cluster,
                double rho, double theta);

std::vector<moveit_msgs::msg::CollisionObject> segmentObjects(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr>
        &cloud_clusters,
    int num_iterations, const std::string &frame_id, int inlier_threshold,
    int hough_radius_bins, int hough_center_bins,
    double ransac_distance_threshold, int ransac_max_iterations,
    int circle_min_cluster_size, int circle_max_clusters,
    double circle_height_tolerance, double circle_curvature_threshold,
    double circle_radius_tolerance, double circle_normal_angle_threshold,
    double circle_cluster_tolerance, int line_min_cluster_size,
    int line_max_clusters, double line_curvature_threshold,
    double line_cluster_tolerance, double line_rho_threshold,
    double line_theta_threshold);

#endif