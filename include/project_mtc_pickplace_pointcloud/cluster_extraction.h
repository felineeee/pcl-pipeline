#ifndef CLUSTER_EXTRACTION_H
#define CLUSTER_EXTRACTION_H

// Include the header that defines PointXYZRGBNormalRSD and LOG_INFO
#include "project_mtc_pickplace_pointcloud/normals_curvature_and_rsd_estimation.h"
#include <algorithm>
#include <limits>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h> // For region growing segmentation
#include <pcl/segmentation/region_growing.h>

std::vector<pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr> extractClusters(
    const pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr &input_cloud,
    unsigned int min_cluster_size, unsigned int max_cluster_size,
    float smoothness_threshold, float curvature_threshold,
    unsigned int nearest_neighbors);

#endif // CLUSTER_EXTRACTION_H