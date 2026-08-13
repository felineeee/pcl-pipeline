#include "project_mtc_pickplace_pointcloud/plane_segmentation.h"

std::tuple<pcl::PointCloud<pcl::PointXYZRGB>::Ptr,
           pcl::PointCloud<pcl::PointXYZRGB>::Ptr, pcl::ModelCoefficients::Ptr>
segmentPlaneAndObjects(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
    bool enable_cropping, double crop_min_x, double crop_max_x,
    double crop_min_y, double crop_max_y, double crop_min_z, double crop_max_z,
    int max_iteration, double distance_threshold, double z_tolerance,
    double angle_tolerance, int min_cluster_size, int max_cluster_size,
    double cluster_tolerance, int normal_estimation_k,
    double plane_segmentation_threshold, double w_inliners, double w_size,
    double w_distance, double w_orientation) {
  LOG_INFO("Starting plane segmentation. Input cloud size: "
           << input_cloud->size() << " points");

  auto support_plane_cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  auto object_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  auto best_plane_model = std::make_shared<pcl::ModelCoefficients>();

  auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*input_cloud, *cloud_filtered, indices);
  LOG_INFO("Remove NaN points. Filtered cloud size: " << cloud_filtered->size()
                                                      << " points");

  auto cleaned_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  cleaned_cloud->points.reserve(cloud_filtered->points.size());
  for (const auto &point : cloud_filtered->points) {
    if (pcl::isFinite(point)) {
      cleaned_cloud->points.push_back(point);
    }
  }
  cleaned_cloud->width = cleaned_cloud->points.size();
  cleaned_cloud->height = 1;
  cleaned_cloud->is_dense = true;

  LOG_INFO("Removed all invalid points. Final cleaned cloud size: "
           << cleaned_cloud->size() << " points");

  if (cleaned_cloud->empty()) {
    LOG_ERROR("All points were invalid. Cannot proceed with segmentation.");

    return std::make_tuple(
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>),
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>),
        pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients));
  }

  LOG_INFO("Starting surface normal estimation");

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normal(
      new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimation;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZRGB>);
  normal_estimation.setInputCloud(cleaned_cloud);
  normal_estimation.setSearchMethod(tree);
  normal_estimation.setKSearch(normal_estimation_k);
  normal_estimation.compute(*cloud_normal);

  LOG_INFO("Finished normal estimation. Computed " << cloud_normal->size()
                                                   << " normals");

  LOG_INFO(
      "Identifying potential horizontal support surfaces (eg. table, shelves)");
  pcl::PointIndices::Ptr horizontal_indices(new pcl::PointIndices);
  for (size_t i = 0; i < cloud_normal->size(); ++i) {
    if (std::abs(cloud_normal->points[i].normal_z) > angle_tolerance) {
      horizontal_indices->indices.push_back(i);
    }
  }
  LOG_INFO("Found " << horizontal_indices->indices.size()
                    << " points likely belonging to horizontal surfaces");

  if (horizontal_indices->indices.empty()) {
    LOG_ERROR("No horizontal surfaces found");
    return std::make_tuple(support_plane_cloud, object_cloud, best_plane_model);
  }
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr horizontal_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  extract.setInputCloud(cleaned_cloud);
  extract.setIndices(horizontal_indices);
  extract.filter(*horizontal_cloud);
  LOG_INFO("Extracted horizontal cloud with " << horizontal_cloud->size()
                                              << " points");

  LOG_INFO("Starting euclidean clustering");
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cluster_tree(
      new pcl::search::KdTree<pcl::PointXYZRGB>);
  cluster_tree->setInputCloud(horizontal_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(max_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(cluster_tree);
  ec.extract(cluster_indices);
  LOG_INFO("Finished clustering. Found " << cluster_indices.size()
                                         << " clusters");

  if (cluster_indices.empty()) {
    LOG_ERROR("No clusters found");
    return std::make_tuple(support_plane_cloud, object_cloud, best_plane_model);
  }

  LOG_INFO("Processing support surface candidate clusters");
  double best_scores = -std::numeric_limits<double>::max();
  bool found_valid_plane = false;

  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    LOG_INFO("Processing clusters " << i + 1 << " of "
                                    << cluster_indices.size());
    const auto &cluster = cluster_indices[i];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::ExtractIndices<pcl::PointXYZRGB> cluster_extract;
    cluster_extract.setInputCloud(horizontal_cloud);
    cluster_extract.setIndices(std::make_shared<pcl::PointIndices>(cluster));
    cluster_extract.filter(*cluster_cloud);
    LOG_INFO(" Cluster size: " << cluster_cloud->size() << " points");

    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iteration);
    seg.setDistanceThreshold(distance_threshold);
    seg.setInputCloud(cluster_cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() > 0) {
      LOG_INFO(" No model plane found for this cluster. Skipping");
      continue;
    }
    LOG_INFO(" Fitted model plane. Inliers: " << inliers->indices.size());
    Eigen::Vector3f plane_normal(coefficients->values[0],
                                 coefficients->values[1],
                                 coefficients->values[2]);

    Eigen::Vector3f up_vector(0, 0, 1);
    double dot_product = plane_normal.dot(up_vector);

    Eigen::Vector4f plane_center;
    pcl::compute3DCentroid(*cluster_cloud, *inliers, plane_center);

    bool is_valid = true;

    if (enable_cropping) {
      is_valid =
          (plane_center[0] >= crop_min_x && plane_center[0] <= crop_max_x &&
           plane_center[1] >= crop_min_y && plane_center[1] <= crop_max_y &&
           plane_center[2] >= crop_min_z && plane_center[2] <= crop_max_z);
      LOG_INFO(" Cropping enabled. Plane center is within bounds: "
               << (is_valid ? "Yes" : "No"));
    }
    if (!is_valid) {
      LOG_INFO(" Plane model invalid");
      LOG_INFO("  Z-center: " << plane_center[2]
                              << " (tolerance: " << z_tolerance << ")");
      LOG_INFO("  Normal-up dot product: " << dot_product << "(tolerance: "
                                           << angle_tolerance << ")");
      LOG_INFO(" Skipping");
      continue;
    }

    double inlier_count = static_cast<double>(inliers->indices.size());
    double inlier_score = inlier_count / cluster_cloud->size();
    double size_score =
        cluster_cloud->size() / static_cast<double>(cleaned_cloud->size());
    double distance_score = 1.0 - (std::abs(plane_center[2] - z_tolerance));
    double orientation_score = dot_product;

    double total_score = w_inliners * inlier_score + w_size * size_score +
                         w_distance * distance_score +
                         w_orientation * orientation_score;

    LOG_INFO(" Plane model score - Inlier: "
             << inlier_score << ", Size: " << size_score << ", Distance: "
             << distance_score << ", Orientation: " << orientation_score
             << ", Total: " << total_score);

    if (total_score > best_scores) {
      best_scores = total_score;
      *best_plane_model = *coefficients;
      found_valid_plane = true;
      LOG_INFO(" New best plane model found. Score: " << total_score);
    }
  }

  LOG_INFO(
      "Finished processing clusters. Best plane model score: " << best_scores);

  if (!found_valid_plane) {
    LOG_ERROR("No valid plane found");
    LOG_ERROR("Please check the z-tolerance ("
              << z_tolerance << ") and angle_tolerance (" << angle_tolerance
              << ") parameters.");

    support_plane_cloud->width = 0;
    support_plane_cloud->height = 1;
    support_plane_cloud->is_dense = true;
    object_cloud->width = 0;
    object_cloud->height = 1;
    object_cloud->is_dense = true;

    best_plane_model->values.resize(4, 0.0);
    return std::make_tuple(support_plane_cloud, object_cloud, best_plane_model);
  }

  LOG_INFO("Extracting support plane and object planes");
  int support_plane_points = 0;
  int object_points = 0;

  for (const auto &point : cleaned_cloud->points) {
    double distance = best_plane_model->values[0] * point.x +
                      best_plane_model->values[1] * point.y +
                      best_plane_model->values[2] * point.z +
                      best_plane_model->values[3];
    if (std::abs(distance) < plane_segmentation_threshold) {
      support_plane_cloud->points.push_back(point);
      support_plane_points++;
    } else if (distance > 0) {
      if (!enable_cropping ||
          (point.x >= crop_min_x && point.x <= crop_max_x &&
           point.y >= crop_min_y && point.y <= crop_max_y &&
           point.z >= crop_min_z && point.z <= crop_max_z)) {
        object_cloud->points.push_back(point);
        object_points++;
      }
    }
  }

  support_plane_cloud->width = support_plane_cloud->points.size();
  support_plane_cloud->height = 1;
  support_plane_cloud->is_dense = true;

  object_cloud->width = object_cloud->points.size();
  object_cloud->height = 1;
  object_cloud->is_dense = true;

  LOG_INFO("Segmentation complete. Support plane size: "
           << support_plane_points
           << " points, Objectcloud size: " << object_points << " points");
  LOG_INFO("Plane model coefficients: A="
           << best_plane_model->values[0]
           << ", B=" << best_plane_model->values[1]
           << ", C=" << best_plane_model->values[2]
           << ", D=" << best_plane_model->values[3]);

  return std::make_tuple(support_plane_cloud, object_cloud, best_plane_model);
}