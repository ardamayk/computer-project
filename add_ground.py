#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningSceneWorld
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import time

class GroundPlaneAdder(Node):
    def __init__(self):
        super().__init__("ground_plane_adder")
        
        # Create client for applying planning scene
        self.planning_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.planning_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Planning scene service not available, waiting...')
        
        # Create publisher for planning scene
        self.planning_scene_publisher = self.create_publisher(
            CollisionObject, 
            '/collision_object',
            10
        )
        
        # Kısa bir bekleme süresi
        time.sleep(1.0)
        
        # Add ground plane
        self.add_ground_plane()
        
    def add_ground_plane(self):
        # Collision object oluştur
        collision_object = CollisionObject()
        collision_object.header.frame_id = "world"
        collision_object.id = "ground_plane"
        
        # Box primitive oluştur
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2.0, 2.0, 0.01]  # x, y, z boyutları
        
        # Pose oluştur
        pose = Pose()
        pose.position.z = -0.0051  # zemine çok yakın koy
        pose.orientation.w = 1.0
        
        # Collision object'e primitive ve pose ekle
        collision_object.primitives = [box]
        collision_object.primitive_poses = [pose]
        collision_object.operation = CollisionObject.ADD
        
        # Collision object'i yayınla
        self.planning_scene_publisher.publish(collision_object)
        self.get_logger().info("Ground plane added to planning scene.")

def main(args=None):
    rclpy.init(args=args)
    ground_plane_adder = GroundPlaneAdder()
    
    try:
        rclpy.spin_once(ground_plane_adder)
    except KeyboardInterrupt:
        pass
    finally:
        ground_plane_adder.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()