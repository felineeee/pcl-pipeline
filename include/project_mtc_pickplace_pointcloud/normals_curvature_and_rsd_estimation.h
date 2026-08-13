#ifndef NORMALS_CURVATURE_AND_RSD_ESTIMATION_H
#define NORMALS_CURVATURE_AND_RSD_ESTIMATION_H

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/common/pca.h>
#include <pcl/features/rsd.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/mlesac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/region_growing.hpp>
#include <pcl/segmentation/region_growing.h>

namespace pcl {
struct EIGEN_ALIGN16 PointXYZRGBNormalRSD {
  PCL_ADD_POINT4D;
  PCL_ADD_RGB;
  PCL_ADD_NORMAL4D;
  float curvature;
  float r_min;
  float r_max;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace pcl

POINT_CLOUD_REGISTER_POINT_STRUCT(
    pcl::PointXYZRGBNormalRSD,
    (float, x, x)(float, y, y)(float, z, z)(float, rgb, rgb)(
        float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z,
                                                              normal_z)(
        float, curvature, curvature)(float, r_min, r_min)(float, r_max, r_max))

template class pcl::search::Search<pcl::PointXYZRGBNormalRSD>;
template class pcl::search::KdTree<pcl::PointXYZRGBNormalRSD>;
template class pcl::RegionGrowing<pcl::PointXYZRGBNormalRSD, pcl::Normal>;

void LOG_INFO(const std::string &message);

pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr estimateNormalsCurvatureAndRSD(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, int k_neighbors,
    double max_plane_error, int max_iterations, int min_boundary_neighbors,
    double rsd_radius);

#endif