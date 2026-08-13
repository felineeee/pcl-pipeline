#ifndef GET_PLANNING_SCENE_CLIENT_H
#define GET_PLANNING_SCENE_CLIENT_H
#include "moveit_msgs/msg/planning_scene_world.hpp"
#include "mycobot_interfaces/srv/get_planning_scene.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <memory>
#include <shape_msgs/msg/plane.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <string>
#include <vector>

class GetPlanningSceneClient : public rclcpp::Node {
public:
  struct PlanningSceneResponse {
    moveit_msgs::msg::PlanningSceneWorld
        scene_world; // The planning scene world information
    sensor_msgs::msg::PointCloud2 full_cloud; // The full point cloud data
    sensor_msgs::msg::Image rgb_image;        // The RGB image data
    std::string target_object_id;             // The ID of the target object
    std::string support_surface_id;           // The ID of the support surface
    bool success; // Flag indicating whether the service call was successful
  };

  GetPlanningSceneClient();

  PlanningSceneResponse
  call_service(const std::string &target_shape,
               const std::vector<double> &target_dimensions);

private:
  rclcpp::Client<mycobot_interfaces::srv::GetPlanningScene>::SharedPtr client_;

  void log_response_info(const PlanningSceneResponse &response);
};

#endif // GET_PLANNING_SCENE_CLIENT_H