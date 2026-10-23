
import os
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from scipy.spatial.transform import Rotation

import numpy as np

from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo
from perception_interfaces.msg import TargetSegmentation, GraspCandidate, GraspCandidates
from cv_bridge import CvBridge


class CGNBridge(Node):

    def __init__(self):
        super().__init__('cgn_bridge')

        self.bridge = CvBridge()

        # Store recent depth frames for timestamp matching
        self.depth_buffer = deque(maxlen=150)

        self.camera_matrix = None

        # Maximum permitted RGB/depth timestamp difference
        self.max_time_diff = 0.03  # 30 milliseconds

        # Output file shared with Contact-GraspNet
        self.output_path = '/shared/cgn/input.npz'

        self.get_logger().info(
            'Contact-GraspNet bridge initialized'
        )

        self.segmentation_sub = self.create_subscription(
            TargetSegmentation,
            '/sam3/target_seg',
            self.segmentation_callback,
            1
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.grasp_pub = self.create_publisher(
            GraspCandidates,
            '/cgn/grasp_candidates',
            1
        )

    def depth_callback(self, msg):

        # Preserve original 16UC1 depth values
        # passthrough tells the conversion function to keep the resulting  image with the exact same encoding as the incoming ROS Image message
        depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )

        # Convert millimeters to meters
        depth_m = depth_image.astype(np.float32) / 1000.0

        # Convert ROS timestamp into seconds
        timestamp = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        # Store timestamp and depth frame
        self.depth_buffer.append(
            (timestamp, depth_m)
        )

    def camera_info_callback(self, msg):

        # Convert the flattened 9-element matrix into 3x3
        if self.camera_matrix is None:
            self.camera_matrix = np.array(
                    msg.k,
                    dtype=np.float32
            ).reshape(3, 3)

    def transform_to_pose(self, transform):
        
        pose = Pose()

        # Translation: meters in the camera frame
        pose.position.x = float(transform[0, 3])
        pose.position.y = float(transform[1, 3])
        pose.position.z = float(transform[2, 3])

        # Convert 3x3 rotation matrix into quaternion (x, y, z, w)
        quaternion = Rotation.from_matrix(
            transform[:3, :3]
        ).as_quat()

        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])

        return pose

    def segmentation_callback(self, msg):

        if self.camera_matrix is None:
            self.get_logger().warn(
                'Waiting for camera intrinsics'
            )
            return

        if not self.depth_buffer:
            self.get_logger().warn(
                'Waiting for depth frames'
            )
            return

        rgb_msg = msg.rgb
        mask_msg = msg.mask

        # Convert SAM3 mask into NumPy array
        mask = self.bridge.imgmsg_to_cv2(
            mask_msg,
            desired_encoding='passthrough'
        )

        rgb = self.bridge.imgmsg_to_cv2(
            rgb_msg,
            desired_encoding='rgb8'
        )

        # Contact-GraspNet segmentation map:
        # 0 = background, 1 = selected object
        segmap = (mask > 0).astype(np.int32)

        mask_timestamp = (
            rgb_msg.header.stamp.sec
            + rgb_msg.header.stamp.nanosec * 1e-9
        )

        # Find depth frame closest to the mask timestamp
        closest_timestamp, closest_depth = min(
            self.depth_buffer,
            key=lambda item: abs(item[0] - mask_timestamp)
        )

        time_diff = abs(
            closest_timestamp - mask_timestamp
        )

        # Reject frames that are too far apart in time
        if time_diff > self.max_time_diff:
            self.get_logger().warn(
                f'No matching depth frame. '
                f'Time difference: {time_diff:.3f}s'
            )
            return

        # Ensure mask and depth have identical dimensions
        if segmap.shape != closest_depth.shape:
            self.get_logger().error(
                'Mask and depth dimensions do not match'
            )
            return

        # Reject empty segmentation masks
        if not np.any(segmap):
            self.get_logger().warn(
                'Segmentation mask is empty'
            )
            return

        # Ensure the shared output directory exists
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Write to a temporary file first so the CGN
        # container never reads a partially written NPZ
        temp_path = self.output_path + '.tmp.npz'

        np.savez_compressed(
            temp_path,
            depth=closest_depth,
            K=self.camera_matrix,
            segmap=segmap,
            rgb=rgb
        )

        # Atomically replace the previous input file
        os.replace(temp_path, self.output_path)

        self.get_logger().info(
            f'Saved {self.output_path} '
            f'(RGB-depth difference: {time_diff * 1000:.1f} ms)'
        )


def main(args=None):

    rclpy.init(args=args)

    node = CGNBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
