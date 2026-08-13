#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <moveit_msgs/msg/planning_scene_world.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <mycobot_interfaces/srv/get_planning_scene.hpp>

#include "project_mtc_pickplace_pointcloud/cluster_extraction.h"
#include "project_mtc_pickplace_pointcloud/object_segmentation.h"
#include "project_mtc_pickplace_pointcloud/plane_segmentation.h"

// TODO: Define the actual PointXYZRGBNormalRSD struct with RSD fields if
// needed. For now, using PointXYZRGBNormal as a base/alias to allow
// compilation.
using PointXYZRGBNormalRSD = pcl::PointXYZRGBNormal;

// Check if mycobot_interfaces is available, otherwise comment out.
// #include <mycobot_interfaces/srv/get_planning_scene.hpp>

class GetPlanningSceneServer : public rclcpp::Node {
public:
  GetPlanningSceneServer(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : Node("get_planning_scene_server", options) {
    declareParameters();
    createSubscriber();
    createService();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
  }

  void handleService(
      const mycobot_interfaces::srv::GetPlanningScene::Request::SharedPtr req,
      mycobot_interfaces::srv::GetPlanningScene::Response::SharedPtr res);

private:
  void declareParameters();
  void createSubscriber();
  void createService();
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void rgbImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

  sensor_msgs::msg::PointCloud2::SharedPtr
  transformPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud_msg,
                      const std::string &target_frame);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  convertToPCL(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud_msg);
  moveit_msgs::msg::CollisionObject createSupportSurfaceObject(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud,
      pcl::ModelCoefficients::Ptr plane_coefficients,
      const std::string &frame_id);

  moveit_msgs::msg::CollisionObject fitShapeToCluster();
  std::string identifyTargetObject(
      const std::vector<moveit_msgs::msg::CollisionObject> &objects,
      const std::string &target_shape,
      const std::vector<double> &target_dimensions);

  moveit_msgs::msg::PlanningSceneWorld assemblePlanningSceneWorld(
      const std::vector<moveit_msgs::msg::CollisionObject> &collision_objects);

  void savePointCloudToPCD(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
                           const std::string &filename);
  moveit_msgs::msg::CollisionObject fitShapeToCluster(
      const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &cluster,
      const std::string &frame_id, int index);

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  std::string point_cloud_topic_;
  std::string rgb_image_topic_;
  std::string target_frame_;

  // Cropping parameters
  bool enable_cropping_;
  double crop_min_x_;
  double crop_max_x_;
  double crop_min_y_;
  double crop_max_y_;
  double crop_min_z_;
  double crop_max_z_;

  // Parameters for plane and object segmentation
  int max_iterations_;
  double distance_threshold_;
  double z_tolerance_;
  double angle_tolerance_;
  double plane_segmentation_threshold_;
  int min_cluster_size_;
  int max_cluster_size_;
  double cluster_tolerance_;
  int normal_estimation_k_;
  double w_inliers_;
  double w_size_;
  double w_distance_;
  double w_orientation_;
  int max_plane_segmentation_iterations_;
  double plane_segmentation_distance_threshold_;

  // Parameters for creating the support plane
  std::string support_surface_name_;
  double min_surface_thickness_;

  // Parameters for normal, curvature, and RSD estimation
  int k_neighbors_;
  double max_plane_error_;
  int max_iterations_normals_;
  int min_boundary_neighbors_;
  double rsd_radius_;
  pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr cloud_with_features_;

  // Parameters for cluster extraction
  int nearest_neighbors_;
  float smoothness_threshold_;
  float curvature_threshold_;

  // Parameters for object segmentation
  int num_iterations_;
  int inlier_threshold_;
  int hough_radius_bins_;
  int hough_center_bins_;
  double ransac_distance_threshold_;
  int ransac_max_iterations_;

  // Parameters for circle filtering
  int circle_min_cluster_size_;
  int circle_max_clusters_;
  double circle_height_tolerance_;
  double circle_curvature_threshold_;
  double circle_radius_tolerance_;
  double circle_normal_angle_threshold_;
  double circle_cluster_tolerance_;

  // Parameters for line filtering
  int line_min_cluster_size_;
  int line_max_clusters_;
  double line_curvature_threshold_;
  double line_cluster_tolerance_;
  double line_rho_threshold_;
  double line_theta_threshold_;

  // Legacy Shape fitting parameters
  int shape_fitting_max_iterations_;
  double shape_fitting_distance_threshold_;
  double shape_fitting_min_radius_;
  double shape_fitting_max_radius_;
  double shape_fitting_normal_distance_weight_;
  double shape_fitting_normal_search_radius_;

  // For output pcd (point cloud) files
  std::string output_directory_;
  std::string debug_pcd_filename_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      point_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;

  // Service
  rclcpp::Service<mycobot_interfaces::srv::GetPlanningScene>::SharedPtr
      service_;

  // Latest data storage
  sensor_msgs::msg::PointCloud2::SharedPtr latest_point_cloud_;
  sensor_msgs::msg::Image::SharedPtr latest_rgb_image_;
};

void GetPlanningSceneServer::declareParameters() {
  // --- Topic and Frame Parameters ---
  point_cloud_topic_ = this->declare_parameter<std::string>(
      "point_cloud_topic", "/camera_head/depth/color/points");
  rgb_image_topic_ = this->declare_parameter<std::string>(
      "rgb_image_topic", "/camera_head/color/image_raw");
  target_frame_ =
      this->declare_parameter<std::string>("target_frame", "base_link");

  // --- Cropping Parameters ---
  enable_cropping_ = this->declare_parameter<bool>("enable_cropping", true);
  crop_min_x_ = this->declare_parameter<double>("crop_min_x", 0.10);
  crop_max_x_ = this->declare_parameter<double>("crop_max_x", 1.1);
  crop_min_y_ = this->declare_parameter<double>("crop_min_y", 0.00);
  crop_max_y_ = this->declare_parameter<double>("crop_max_y", 0.90);
  crop_min_z_ = this->declare_parameter<double>(
      "crop_min_z", -std::numeric_limits<double>::infinity());
  crop_max_z_ = this->declare_parameter<double>(
      "crop_max_z", std::numeric_limits<double>::infinity());

  // --- Plane and Object Segmentation ---
  max_iterations_ = this->declare_parameter<int>("max_iterations", 100);
  distance_threshold_ =
      this->declare_parameter<double>("distance_threshold", 0.01);
  z_tolerance_ = this->declare_parameter<double>("z_tolerance", 0.03);
  angle_tolerance_ =
      this->declare_parameter<double>("angle_tolerance", 0.9990482216);
  plane_segmentation_threshold_ =
      this->declare_parameter<double>("plane_segmentation_threshold", 0.001);
  min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 100);
  max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 1000000);
  cluster_tolerance_ =
      this->declare_parameter<double>("cluster_tolerance", 0.02);
  normal_estimation_k_ =
      this->declare_parameter<int>("normal_estimation_k", 30);
  w_inliers_ = this->declare_parameter<double>("w_inliers", 1.0);
  w_size_ = this->declare_parameter<double>("w_size", 1.0);
  w_distance_ = this->declare_parameter<double>("w_distance", 1.0);
  w_orientation_ = this->declare_parameter<double>("w_orientation", 1.0);
  max_plane_segmentation_iterations_ =
      this->declare_parameter<int>("max_plane_segmentation_iterations", 1000);
  plane_segmentation_distance_threshold_ = this->declare_parameter<double>(
      "plane_segmentation_distance_threshold", 0.01);

  // --- Support Surface Parameters ---
  support_surface_name_ = this->declare_parameter<std::string>(
      "support_surface_name", "support_surface");
  min_surface_thickness_ =
      this->declare_parameter<double>("min_surface_thickness", 0.0001);

  // --- Normal and RSD Estimation ---
  k_neighbors_ = this->declare_parameter<int>("k_neighbors", 30);
  max_plane_error_ = this->declare_parameter<double>("max_plane_error", 0.01);
  max_iterations_normals_ =
      this->declare_parameter<int>("max_iterations_normals", 100);
  min_boundary_neighbors_ =
      this->declare_parameter<int>("min_boundary_neighbors", 10);
  rsd_radius_ = this->declare_parameter<double>("rsd_radius", 0.01);

  // --- Cluster Extraction (Region Growing) ---
  nearest_neighbors_ = this->declare_parameter<int>("nearest_neighbors", 30);
  smoothness_threshold_ =
      this->declare_parameter<float>("smoothness_threshold", 20.0f);
  curvature_threshold_ =
      this->declare_parameter<float>("curvature_threshold", 0.2f);

  // --- Object Segmentation (Hough/RANSAC) ---
  num_iterations_ = this->declare_parameter<int>("num_iterations", 5);
  inlier_threshold_ = this->declare_parameter<int>("inlier_threshold", 85);
  hough_radius_bins_ = this->declare_parameter<int>("hough_radius_bins", 50);
  hough_center_bins_ = this->declare_parameter<int>("hough_center_bins", 50);
  ransac_distance_threshold_ =
      this->declare_parameter<double>("ransac_distance_threshold", 0.001);
  ransac_max_iterations_ =
      this->declare_parameter<int>("ransac_max_iterations", 1000);

  // --- Circle Filtering ---
  circle_min_cluster_size_ =
      this->declare_parameter<int>("circle_min_cluster_size", 20);
  circle_max_clusters_ = this->declare_parameter<int>("circle_max_clusters", 2);
  circle_height_tolerance_ =
      this->declare_parameter<double>("circle_height_tolerance", 0.025);
  circle_curvature_threshold_ =
      this->declare_parameter<double>("circle_curvature_threshold", 0.0011);
  circle_radius_tolerance_ =
      this->declare_parameter<double>("circle_radius_tolerance", 0.020);
  circle_normal_angle_threshold_ =
      this->declare_parameter<double>("circle_normal_angle_threshold", 0.2);
  circle_cluster_tolerance_ =
      this->declare_parameter<double>("circle_cluster_tolerance", 0.025);

  // --- Line Filtering ---
  line_min_cluster_size_ =
      this->declare_parameter<int>("line_min_cluster_size", 20);
  line_max_clusters_ = this->declare_parameter<int>("line_max_clusters", 1);
  line_curvature_threshold_ =
      this->declare_parameter<double>("line_curvature_threshold", 0.0011);
  line_cluster_tolerance_ =
      this->declare_parameter<double>("line_cluster_tolerance", 0.025);
  line_rho_threshold_ =
      this->declare_parameter<double>("line_rho_threshold", 0.01);
  line_theta_threshold_ =
      this->declare_parameter<double>("line_theta_threshold", 0.1);

  // --- Legacy Shape Fitting ---
  shape_fitting_max_iterations_ =
      this->declare_parameter<int>("shape_fitting_max_iterations", 1000);
  shape_fitting_distance_threshold_ =
      this->declare_parameter<double>("shape_fitting_distance_threshold", 0.01);
  shape_fitting_min_radius_ =
      this->declare_parameter<double>("shape_fitting_min_radius", 0.01);
  shape_fitting_max_radius_ =
      this->declare_parameter<double>("shape_fitting_max_radius", 0.1);
  shape_fitting_normal_distance_weight_ = this->declare_parameter<double>(
      "shape_fitting_normal_distance_weight", 0.1);
  shape_fitting_normal_search_radius_ = this->declare_parameter<double>(
      "shape_fitting_normal_search_radius", 0.05);

  // --- Debug/Output ---
  output_directory_ =
      this->declare_parameter<std::string>("output_directory", "/tmp/");
  debug_pcd_filename_ = this->declare_parameter<std::string>(
      "debug_pcd_filename", "debug_cloud.pcd");
}

void GetPlanningSceneServer::createSubscriber() {
  point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      point_cloud_topic_, 10,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pointCloudCallback(msg);
      });
  rgb_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      rgb_image_topic_, 10,
      [this](const sensor_msgs::msg::Image::SharedPtr msg) {
        rgbImageCallback(msg);
      });

  RCLCPP_INFO(this->get_logger(), "Subscribed to point cloud topic: %s",
              point_cloud_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Subscribed to RGB image topic: %s",
              rgb_image_topic_.c_str());
}
void GetPlanningSceneServer::createService() {
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  service_ = this->create_service<mycobot_interfaces::srv::GetPlanningScene>(
      "get_planning_scene",
      [this](
          const mycobot_interfaces::srv::GetPlanningScene::Request::SharedPtr
              req,
          mycobot_interfaces::srv::GetPlanningScene::Response::SharedPtr res) {
        handleService(req, res);
      },
      qos);
}

void GetPlanningSceneServer::handleService(
    const mycobot_interfaces::srv::GetPlanningScene::Request::SharedPtr req,
    mycobot_interfaces::srv::GetPlanningScene::Response::SharedPtr res) {
  res->success = false;

  if (!latest_point_cloud_ || !latest_rgb_image_) {
    RCLCPP_WARN(this->get_logger(), "No point cloud or RGB image available.");
    return;
  }
  if (req->target_shape.empty() || req->target_dimensions.empty()) {
    RCLCPP_WARN(this->get_logger(), "Invalid target shape or dimensions.");
    return;
  }
  std::string original_cloud_frame = latest_point_cloud_->header.frame_id;

  auto transformed_cloud =
      transformPointCloud(latest_point_cloud_, target_frame_);
  if (!transformed_cloud) {
    RCLCPP_WARN(this->get_logger(), "Failed to transform point cloud.");
    return;
  }

  // auto pcl_cloud = convertToPCL(transformed_cloud);
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud =
      convertToPCL(transformed_cloud);

  if (!pcl_cloud) {
    RCLCPP_WARN(this->get_logger(), "Failed to convert point cloud to PCL.");
    return;
  }

  savePointCloudToPCD(pcl_cloud, "4_convertToPCL_" + debug_pcd_filename_);

  // plane_segmentation.h
  auto [support_plane_cloud_, objects_cloud_, plane_coefficients_] =
      segmentPlaneAndObjects(pcl_cloud, enable_cropping_, crop_min_x_,
                             crop_max_x_, crop_min_y_, crop_max_y_, crop_min_z_,
                             crop_max_z_, max_plane_segmentation_iterations_,
                             plane_segmentation_distance_threshold_,
                             z_tolerance_, angle_tolerance_, min_cluster_size_,
                             max_cluster_size_, cluster_tolerance_,
                             normal_estimation_k_,
                             plane_segmentation_threshold_, w_inliers_, w_size_,
                             w_distance_, w_orientation_);

  if (!support_plane_cloud_ || !objects_cloud_ || !plane_coefficients_) {
    RCLCPP_WARN(this->get_logger(), "Failed to segment plane and objects.");
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Plane and object segmentation successful");
  RCLCPP_INFO(this->get_logger(), "Support plane cloud size %zu",
              support_plane_cloud_->size());
  RCLCPP_INFO(this->get_logger(), "Plane coefficients [%.3f, %.3f, %.3f, %.3f]",
              plane_coefficients_->values[0], plane_coefficients_->values[1],
              plane_coefficients_->values[2], plane_coefficients_->values[3]);
  RCLCPP_INFO(this->get_logger(), "Objects cloud size %zu",
              objects_cloud_->size());

  savePointCloudToPCD(support_plane_cloud_,
                      "5_support_plane" + debug_pcd_filename_);
  savePointCloudToPCD(objects_cloud_, "6_objects" + debug_pcd_filename_);

  moveit_msgs::msg::CollisionObject support_surface =
      createSupportSurfaceObject(support_plane_cloud_, plane_coefficients_,
                                 target_frame_);

  if (support_surface.id.empty()) {
    RCLCPP_WARN(this->get_logger(), "Support surface collision object creation "
                                    "failed or resulted in an invalid object");
  } else {
    RCLCPP_INFO(
        this->get_logger(),
        "Adding support surface collision object to the planning scene");
    res->scene_world.collision_objects.push_back(support_surface);
    res->support_surface_id = support_surface.id;
  }

  cloud_with_features_ = estimateNormalsCurvatureAndRSD(
      objects_cloud_, k_neighbors_, max_plane_error_, max_iterations_normals_,
      min_boundary_neighbors_, rsd_radius_);

  if (!cloud_with_features_ || cloud_with_features_->empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to estimate normals, curvature, and RSD");
    return;
  }
  RCLCPP_INFO(
      this->get_logger(),
      "Successfully estimated normals, curvature, and RSD for %zu points",
      cloud_with_features_->size());
  // cluster_extraction.h
  std::vector<pcl::PointCloud<pcl::PointXYZRGBNormalRSD>::Ptr> clusters =
      extractClusters(cloud_with_features_, min_cluster_size_,
                      max_cluster_size_, smoothness_threshold_,
                      curvature_threshold_, nearest_neighbors_);

  if (clusters.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Node %s failed to extract any clusters from the point cloud",
                 this->get_name());
    return;
  }
  RCLCPP_INFO(
      this->get_logger(),
      "Node '%s' successfully extracted %zu clusters from the point cloud",
      this->get_name(), clusters.size());

  // object_segmentation.h
  std::vector<moveit_msgs::msg::CollisionObject> segmented_objects =
      segmentObjects(
          clusters, num_iterations_, target_frame_, inlier_threshold_,
          hough_radius_bins_, hough_center_bins_, ransac_distance_threshold_,
          ransac_max_iterations_, circle_min_cluster_size_,
          circle_max_clusters_, circle_height_tolerance_,
          circle_curvature_threshold_, circle_radius_tolerance_,
          circle_normal_angle_threshold_, circle_cluster_tolerance_,
          line_min_cluster_size_, line_max_clusters_, line_curvature_threshold_,
          line_cluster_tolerance_, line_rho_threshold_, line_theta_threshold_);

  RCLCPP_INFO(this->get_logger(),
              "Segmented %zu objects from the point cloud clusters",
              segmented_objects.size());
  for (const auto &obj : segmented_objects) {
    res->scene_world.collision_objects.push_back(obj);
  }

  std::string target_object_id =
      identifyTargetObject(res->scene_world.collision_objects,
                           req->target_shape, req->target_dimensions);

  if (!target_object_id.empty()) {
    res->target_object_id = target_object_id;
  }

  try {
    res->scene_world =
        assemblePlanningSceneWorld(res->scene_world.collision_objects);
    if (res->scene_world.collision_objects.empty()) {
      RCLCPP_ERROR(this->get_logger(),
                   "Failed to assemble planning scene world");
      res->success = false;
      return;
    }
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to assemble planning scene world: %s", e.what());
    res->success = false;
    return;
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(),
                 "Unknown error occurred while assembling PlanningSceneWorld");
    return;
  }

  res->full_cloud = *latest_point_cloud_;
  res->rgb_image = *latest_rgb_image_;
  res->target_object_id = target_object_id;

  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Success %s",
              res->success ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Target object ID: %s",
              res->target_object_id.c_str());
  RCLCPP_INFO(this->get_logger(), "Support surface ID: %s",
              res->support_surface_id.c_str());

  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Full cloud frame ID: %s",
              res->full_cloud.header.frame_id.c_str());
  RCLCPP_INFO(this->get_logger(), "Full cloud size: %d x %d",
              res->full_cloud.width, res->full_cloud.height);

  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Full cloud frame ID: %s",
              res->rgb_image.header.frame_id.c_str());
  RCLCPP_INFO(this->get_logger(), "RGB image size: %d x %d",
              res->rgb_image.width, res->rgb_image.height);

  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Number of collision objects: %zu",
              res->scene_world.collision_objects.size());
  for (const auto &obj : res->scene_world.collision_objects) {
    if (!obj.primitives.empty()) {
      const auto &primitive = obj.primitives[0];
      const auto &pose = obj.primitive_poses[0];
      std::string type_str;
      std::string dimension_str;

      switch (primitive.type) {
      case shape_msgs::msg::SolidPrimitive::BOX:
        type_str = "BOX";
        dimension_str = "x=" + std::to_string(primitive.dimensions[0]) +
                        ", y=" + std::to_string(primitive.dimensions[1]) +
                        ", z=" + std::to_string(primitive.dimensions[2]);
        break;
      case shape_msgs::msg::SolidPrimitive::CYLINDER:
        type_str = "CYLINDER";
        dimension_str =
            "radius=" +
            std::to_string(
                primitive.dimensions
                    [shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS]) +
            ", height=" +
            std::to_string(
                primitive.dimensions
                    [shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT]);
        break;
      default:
        continue;
      }
      RCLCPP_INFO(this->get_logger(), " ");
      RCLCPP_INFO(this->get_logger(),
                  "Collision Object: ID=%s Frame=%s Type=%s", obj.id.c_str(),
                  obj.header.frame_id.c_str(), type_str.c_str());
      RCLCPP_INFO(this->get_logger(), " Position: x=%.4f, y=%.4f, z=%.4f",
                  pose.position.x, pose.position.y, pose.position.z);
      RCLCPP_INFO(this->get_logger(),
                  " Orientation: x=%.4f, y=%.4f, z=%.4f, w=%.4f",
                  pose.orientation.x, pose.orientation.y, pose.orientation.z,
                  pose.orientation.w);
      RCLCPP_INFO(this->get_logger(), " Dimensions %s", dimension_str.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), " ");
      RCLCPP_WARN(this->get_logger(),
                  "Collision Object: ID=%s has no primitives", obj.id.c_str());
    }
  }
  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Original point cloud frame: %s",
              latest_point_cloud_->header.frame_id.c_str());
  RCLCPP_INFO(this->get_logger(), "Target frame used for processing: %s",
              target_frame_.c_str());

  if (!res->scene_world.collision_objects.empty() &&
      !res->full_cloud.data.empty() && !res->rgb_image.data.empty() &&
      !res->target_object_id.empty() && !res->support_surface_id.empty()) {
    res->success = true;
    RCLCPP_INFO(this->get_logger(),
                "===== Service response filled successfully =====");
  } else {
    if (res->scene_world.collision_objects.empty()) {
      RCLCPP_WARN(this->get_logger(), "No collision objects found.");
    }
    if (res->full_cloud.data.empty()) {
      RCLCPP_WARN(this->get_logger(), "Full cloud data is empty.");
    }
    if (res->rgb_image.data.empty()) {
      RCLCPP_WARN(this->get_logger(), "RGB image data is empty.");
    }
    if (res->target_object_id.empty()) {
      RCLCPP_WARN(this->get_logger(), "Target object ID is empty.");
    }
    if (res->support_surface_id.empty()) {
      RCLCPP_WARN(this->get_logger(), "Support surface ID is empty.");
    }
  }
  RCLCPP_INFO(this->get_logger(), " ");
  RCLCPP_INFO(this->get_logger(), "Service response logging completed!");
}
void GetPlanningSceneServer::pointCloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  if (msg != nullptr && !msg->data.empty()) {
    latest_point_cloud_ = msg;
  }
}

void GetPlanningSceneServer::rgbImageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  if (msg != nullptr && !msg->data.empty()) {
    latest_rgb_image_ = msg;
  }
}

sensor_msgs::msg::PointCloud2::SharedPtr
GetPlanningSceneServer::transformPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr &cloud_msg,
    const std::string &target_frame) {
  RCLCPP_INFO(this->get_logger(), "Transforming point cloud from %s to %s",
              cloud_msg->header.frame_id.c_str(), target_frame.c_str());

  if (cloud_msg->header.frame_id == target_frame) {
    RCLCPP_INFO(this->get_logger(),
                "Point cloud is already in the target frame");
  }

  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    // Look up the transformation from the cloud's frame to the target frame
    transform_stamped = tf_buffer_->lookupTransform(
        target_frame, cloud_msg->header.frame_id, tf2::TimePointZero);
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(this->get_logger(), "Could not transform point cloud: %s",
                 ex.what());
    return nullptr;
  }

  // Convert ROS PointCloud2 to PCL PointCloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

  // Create transformation matrix from transform_stamped
  Eigen::Affine3d transform_eigen;
  transform_eigen = tf2::transformToEigen(transform_stamped);

  // Transform the PCL cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud(*pcl_cloud, *transformed_cloud, transform_eigen);

  if (enable_cropping_) {
    RCLCPP_INFO(this->get_logger(),
                "Cropping is enabled. Applying crop box filter.");

    // Crop the transformed cloud
    pcl::CropBox<pcl::PointXYZRGB> crop_box;
    crop_box.setInputCloud(transformed_cloud);

    crop_box.setMin(
        Eigen::Vector4f(crop_min_x_, crop_min_y_, crop_min_z_, 1.0));
    crop_box.setMax(
        Eigen::Vector4f(crop_max_x_, crop_max_y_, crop_max_z_, 1.0));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    crop_box.filter(*cropped_cloud);

    transformed_cloud = cropped_cloud;
    RCLCPP_INFO(this->get_logger(), "Point cloud cropped. New size: %zu points",
                transformed_cloud->size());
  } else {
    RCLCPP_INFO(this->get_logger(),
                "Cropping is disabled. Using full transformed point cloud.");
  }

  // Convert back to ROS PointCloud2
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_out(
      new sensor_msgs::msg::PointCloud2);
  pcl::toROSMsg(*transformed_cloud, *cloud_out);

  // Update the frame_id and timestamp of the transformed cloud
  cloud_out->header.frame_id = target_frame;
  cloud_out->header.stamp = this->now();

  RCLCPP_INFO(this->get_logger(), "Point cloud transformed successfully");
  return cloud_out;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr GetPlanningSceneServer::convertToPCL(
    const sensor_msgs::msg::PointCloud2::SharedPtr &cloud_msg) {
  RCLCPP_INFO(this->get_logger(), "Converting PointCloud2 to PCL PointCloud");

  auto pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  try {
    if (!cloud_msg) {
      throw std::runtime_error("Input PointCloud2 message is null");
    }
    if (cloud_msg->data.empty()) {
      throw std::runtime_error("Input PointCloud2 message has no data");
    }

    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

    if (pcl_cloud->empty()) {
      throw std::runtime_error("Resulting PCL cloud is empty after conversion");
    }

    RCLCPP_INFO(
        this->get_logger(),
        "PointCloud2 successfully converted to PCL PointCloud with %zu points",
        pcl_cloud->size());
  } catch (const pcl::PCLException &e) {
    RCLCPP_ERROR(this->get_logger(), "PCL error in convertToPCL: %s", e.what());
    return nullptr;
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Error in convertToPCL: %s", e.what());
    return nullptr;
  } catch (...) {
    RCLCPP_ERROR(this->get_logger(), "Unknown error occurred in convertToPCL");
    return nullptr;
  }

  return pcl_cloud;
}

moveit_msgs::msg::CollisionObject
GetPlanningSceneServer::createSupportSurfaceObject(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud,
    pcl::ModelCoefficients::Ptr plane_coefficients,
    const std::string &frame_id) {
  RCLCPP_INFO(this->get_logger(), "Creating support surface object");

  moveit_msgs::msg::CollisionObject support_surface;

  try {
    // Validate input parameters
    if (!plane_coefficients || plane_coefficients->values.size() != 4) {
      throw std::invalid_argument("Invalid plane coefficients");
    }
    if (frame_id.empty()) {
      throw std::invalid_argument("Empty frame_id");
    }
    if (!plane_cloud || plane_cloud->empty()) {
      throw std::invalid_argument("Invalid or empty plane point cloud");
    }

    // Calculate bounding box of the plane cloud
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*plane_cloud, min_pt, max_pt);

    // If cropping is enabled, limit the bounding box to the crop box limits
    if (enable_cropping_) {
      min_pt[0] = std::max(min_pt[0], static_cast<float>(crop_min_x_));
      min_pt[1] = std::max(min_pt[1], static_cast<float>(crop_min_y_));
      min_pt[2] = std::max(min_pt[2], static_cast<float>(crop_min_z_));
      max_pt[0] = std::min(max_pt[0], static_cast<float>(crop_max_x_));
      max_pt[1] = std::min(max_pt[1], static_cast<float>(crop_max_y_));
      max_pt[2] = std::min(max_pt[2], static_cast<float>(crop_max_z_));
    }

    // Calculate centroid
    Eigen::Vector4f centroid;
    centroid = (min_pt + max_pt) / 2.0f;

    // Create a box primitive
    shape_msgs::msg::SolidPrimitive box_primitive;
    box_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    box_primitive.dimensions.resize(3);
    box_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] =
        max_pt[0] - min_pt[0];
    box_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] =
        max_pt[1] - min_pt[1];
    box_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] =
        max_pt[2] - min_pt[2];

    // Ensure a minimum thickness
    if (box_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] <
        min_surface_thickness_) {
      box_primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] =
          min_surface_thickness_;
    }

    // Calculate an appropriate pose for the box based on the plane's normal
    // vector
    geometry_msgs::msg::Pose box_pose;
    Eigen::Vector3d normal(plane_coefficients->values[0],
                           plane_coefficients->values[1],
                           plane_coefficients->values[2]);
    normal.normalize();

    // Calculate rotation to align the plane normal with the z-axis
    Eigen::Quaterniond rotation;
    rotation.setFromTwoVectors(Eigen::Vector3d::UnitZ(), normal);

    // Set the pose
    box_pose.position.x = centroid[0];
    box_pose.position.y = centroid[1];
    box_pose.position.z = centroid[2];
    box_pose.orientation.x = rotation.x();
    box_pose.orientation.y = rotation.y();
    box_pose.orientation.z = rotation.z();
    box_pose.orientation.w = rotation.w();

    RCLCPP_INFO(this->get_logger(),
                "Support surface box dimensions: [%.4f, %.4f, %.4f]",
                box_primitive.dimensions[0], box_primitive.dimensions[1],
                box_primitive.dimensions[2]);
    RCLCPP_INFO(this->get_logger(),
                "Support surface orientation: [%.4f, %.4f, %.4f, %.4f]",
                box_pose.orientation.x, box_pose.orientation.y,
                box_pose.orientation.z, box_pose.orientation.w);

    // Set up the CollisionObject
    support_surface.header.frame_id = frame_id;
    support_surface.header.stamp = this->now();
    support_surface.id = support_surface_name_;
    support_surface.primitives.push_back(box_primitive);
    support_surface.primitive_poses.push_back(box_pose);
    support_surface.operation = moveit_msgs::msg::CollisionObject::ADD;

    RCLCPP_INFO(this->get_logger(),
                "Support surface object created successfully as a box");
    RCLCPP_INFO(this->get_logger(),
                "Support surface normal: [%.4f, %.4f, %.4f]", normal.x(),
                normal.y(), normal.z());
    RCLCPP_INFO(this->get_logger(),
                "Support surface position: [%.4f, %.4f, %.4f]",
                box_pose.position.x, box_pose.position.y, box_pose.position.z);
    RCLCPP_INFO(this->get_logger(),
                "Support surface orientation: [%.4f, %.4f, %.4f, %.4f]",
                box_pose.orientation.x, box_pose.orientation.y,
                box_pose.orientation.z, box_pose.orientation.w);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(),
                 "Error creating support surface object: %s", e.what());
  } catch (...) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Unknown error occurred while creating support surface object");
  }
  return support_surface;
}

moveit_msgs::msg::CollisionObject GetPlanningSceneServer::fitShapeToCluster(
    const std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> &cluster,
    const std::string &frame_id, int index) {
  RCLCPP_INFO(this->get_logger(), "Fitting shape to cluster %d", index);

  moveit_msgs::msg::CollisionObject collision_object;
  collision_object.header.frame_id = frame_id;
  collision_object.id = "object_" + std::to_string(index);

  if (cluster->empty()) {
    RCLCPP_WARN(this->get_logger(), "Empty cluster. Skipping shape fitting.");
    return collision_object;
  }

  // Structure to store fitting results
  struct FitResult {
    std::string shape_type;
    std::shared_ptr<pcl::ModelCoefficients> coefficients;
    std::shared_ptr<pcl::PointIndices> inliers;
    double fitness_score;
  };

  std::vector<FitResult> fit_results;

  // Attempt to fit multiple primitive shapes using RANSAC
  // Box fitting
  {
    auto seg = std::make_shared<pcl::SACSegmentation<pcl::PointXYZRGB>>();
    seg->setOptimizeCoefficients(true);
    seg->setModelType(pcl::SACMODEL_PARALLEL_PLANE);
    seg->setMethodType(pcl::SAC_RANSAC);
    seg->setMaxIterations(shape_fitting_max_iterations_);
    seg->setDistanceThreshold(shape_fitting_distance_threshold_);

    FitResult box_fit;
    box_fit.shape_type = "box";
    box_fit.coefficients = std::make_shared<pcl::ModelCoefficients>();
    box_fit.inliers = std::make_shared<pcl::PointIndices>();

    seg->setInputCloud(cluster);
    seg->segment(*(box_fit.inliers), *(box_fit.coefficients));

    if (box_fit.inliers->indices.size() > 0) {
      auto box_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
      pcl::copyPointCloud(*cluster, box_fit.inliers->indices, *box_cloud);
      box_fit.fitness_score =
          static_cast<double>(box_fit.inliers->indices.size()) /
          cluster->size();
      fit_results.push_back(box_fit);
    }
  }

  // Cylinder fitting
  {
    auto seg = std::make_shared<
        pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal>>();
    seg->setOptimizeCoefficients(true);
    seg->setModelType(pcl::SACMODEL_CYLINDER);
    seg->setMethodType(pcl::SAC_RANSAC);
    seg->setMaxIterations(shape_fitting_max_iterations_);
    seg->setDistanceThreshold(shape_fitting_distance_threshold_);
    seg->setRadiusLimits(shape_fitting_min_radius_, shape_fitting_max_radius_);
    seg->setNormalDistanceWeight(shape_fitting_normal_distance_weight_);

    auto ne = std::make_shared<
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal>>();
    auto cloud_normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
    ne->setSearchMethod(tree);
    ne->setInputCloud(cluster);
    ne->setRadiusSearch(shape_fitting_normal_search_radius_);
    ne->compute(*cloud_normals);

    seg->setInputCloud(cluster);
    seg->setInputNormals(cloud_normals);

    FitResult cylinder_fit;
    cylinder_fit.shape_type = "cylinder";
    cylinder_fit.coefficients = std::make_shared<pcl::ModelCoefficients>();
    cylinder_fit.inliers = std::make_shared<pcl::PointIndices>();

    seg->segment(*(cylinder_fit.inliers), *(cylinder_fit.coefficients));

    if (cylinder_fit.inliers->indices.size() > 0) {
      cylinder_fit.fitness_score =
          static_cast<double>(cylinder_fit.inliers->indices.size()) /
          cluster->size();
      fit_results.push_back(cylinder_fit);
    }
  }

  // Compare fitted shapes and select the best fit
  FitResult best_fit;
  double best_score = 0.0;
  for (const auto &fit : fit_results) {
    if (fit.fitness_score > best_score) {
      best_score = fit.fitness_score;
      best_fit = fit;
    }
  }

  if (best_fit.shape_type.empty()) {
    RCLCPP_WARN(this->get_logger(), "No shape could be fitted to the cluster.");
    return collision_object;
  }

  // Create CollisionObject for the best-fitting shape
  shape_msgs::msg::SolidPrimitive primitive;
  geometry_msgs::msg::Pose pose;

  if (best_fit.shape_type == "box") {
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions.resize(3);
    // Calculate box dimensions from point cloud
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X] =
        max_pt[0] - min_pt[0];
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y] =
        max_pt[1] - min_pt[1];
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z] =
        max_pt[2] - min_pt[2];
    pose.position.x = (min_pt[0] + max_pt[0]) / 2;
    pose.position.y = (min_pt[1] + max_pt[1]) / 2;
    pose.position.z = (min_pt[2] + max_pt[2]) / 2;
  } else if (best_fit.shape_type == "cylinder") {
    primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    primitive.dimensions.resize(2);
    Eigen::Vector3f axis(best_fit.coefficients->values[3],
                         best_fit.coefficients->values[4],
                         best_fit.coefficients->values[5]);
    Eigen::Vector3f center(best_fit.coefficients->values[0],
                           best_fit.coefficients->values[1],
                           best_fit.coefficients->values[2]);
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] =
        best_fit.coefficients->values[6];

    // Calculate cylinder height
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    Eigen::Vector3f min_vec = min_pt.head<3>();
    Eigen::Vector3f max_vec = max_pt.head<3>();
    Eigen::Vector3f diff = max_vec - min_vec;
    primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] =
        diff.dot(axis.normalized());

    Eigen::Quaternionf quat =
        Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), axis);
    pose.position.x = center[0];
    pose.position.y = center[1];
    pose.position.z = center[2];
    pose.orientation.x = quat.x();
    pose.orientation.y = quat.y();
    pose.orientation.z = quat.z();
    pose.orientation.w = quat.w();
  }

  collision_object.id = best_fit.shape_type + "_" + std::to_string(index);
  collision_object.primitives.push_back(primitive);
  collision_object.primitive_poses.push_back(pose);
  collision_object.operation = moveit_msgs::msg::CollisionObject::ADD;

  RCLCPP_INFO(this->get_logger(), "Fitted %s to cluster %d with score %.2f",
              best_fit.shape_type.c_str(), index, best_score);

  return collision_object;
}

std::string GetPlanningSceneServer::identifyTargetObject(
    const std::vector<moveit_msgs::msg::CollisionObject> &objects,
    const std::string &target_shape,
    const std::vector<double> &target_dimensions) {

  double best_score = 0.0;
  std::string best_match_id;

  for (const auto &object : objects) {
    if (object.primitives.empty())
      continue;

    const auto &primitive = object.primitives[0];
    double shape_score = 0.0;
    double dimension_score = 0.0;

    // Compare shape types
    if ((target_shape == "cylinder" &&
         primitive.type == shape_msgs::msg::SolidPrimitive::CYLINDER) ||
        (target_shape == "box" &&
         primitive.type == shape_msgs::msg::SolidPrimitive::BOX)) {
      shape_score = 1.0;
    } else {
      continue; // Skip to next object if shape doesn't match
    }

    // Compare dimensions
    std::vector<double> object_dimensions;
    switch (primitive.type) {
    case shape_msgs::msg::SolidPrimitive::CYLINDER:
      object_dimensions = {primitive.dimensions[primitive.CYLINDER_HEIGHT],
                           primitive.dimensions[primitive.CYLINDER_RADIUS]};
      break;
    case shape_msgs::msg::SolidPrimitive::BOX:
      object_dimensions = {primitive.dimensions[primitive.BOX_X],
                           primitive.dimensions[primitive.BOX_Y],
                           primitive.dimensions[primitive.BOX_Z]};
      break;
    default:
      continue; // Skip to next object if shape is neither cylinder nor box
    }

    // Calculate dimension similarity score
    if (object_dimensions.size() == target_dimensions.size()) {
      double total_diff = 0.0;
      for (size_t i = 0; i < object_dimensions.size(); ++i) {
        double diff = std::abs(object_dimensions[i] - target_dimensions[i]);
        total_diff += diff / target_dimensions[i]; // Normalize the difference
      }
      dimension_score =
          1.0 - (total_diff /
                 object_dimensions.size()); // Average normalized similarity
      dimension_score =
          std::max(0.0, dimension_score); // Ensure non-negative score
    }

    // Calculate overall similarity score
    double similarity_score = 0.7 * shape_score + 0.3 * dimension_score;

    // Update best match if this object has a higher similarity score
    if (similarity_score > best_score) {
      best_score = similarity_score;
      best_match_id = object.id;
    }
  }

  if (!best_match_id.empty()) {
    RCLCPP_INFO(this->get_logger(),
                "Best matching object found: %s with similarity score: %.2f",
                best_match_id.c_str(), best_score);
  } else {
    RCLCPP_WARN(this->get_logger(),
                "No matching object found for the target shape and dimensions");
  }

  return best_match_id;
}

moveit_msgs::msg::PlanningSceneWorld
GetPlanningSceneServer::assemblePlanningSceneWorld(
    const std::vector<moveit_msgs::msg::CollisionObject> &collision_objects) {

  RCLCPP_INFO(this->get_logger(), "Assembling PlanningSceneWorld");

  moveit_msgs::msg::PlanningSceneWorld planning_scene_world;

  for (const auto &object : collision_objects) {
    planning_scene_world.collision_objects.push_back(object);

    if (object.header.frame_id != target_frame_) {
      RCLCPP_WARN(
          this->get_logger(),
          "CollisionObject '%s' is in frame '%s', not in target frame '%s'. "
          "This may cause issues in planning.",
          object.id.c_str(), object.header.frame_id.c_str(),
          target_frame_.c_str());
    }
  }

  RCLCPP_INFO(
      this->get_logger(),
      "Assembled PlanningSceneWorld successfully with %zu collision objects",
      planning_scene_world.collision_objects.size());

  return planning_scene_world;
}

// Used for debugging to see the point cloud at interim steps
void GetPlanningSceneServer::savePointCloudToPCD(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud,
    const std::string &filename) {
  std::string full_path = output_directory_ + filename;
  if (pcl::io::savePCDFileBinary(full_path, *cloud) == -1) {
    RCLCPP_ERROR(this->get_logger(), "Failed to save %s", full_path.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Saved %s with %zu points.",
                full_path.c_str(), cloud->size());
  }
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GetPlanningSceneServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
