from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # Path to the RealSense package's standard launch file
    realsense_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # Launch the RealSense camera
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments={
            'align_depth.enable': 'true',
            'enable_sync': 'true'
        }.items()
    )

    # Allow the target object to be specified when launching
    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='bottle'
    )
    
    target_object = LaunchConfiguration('target_object')

    # Launch our SAM3 ROS node
    sam3_node = Node(
        package='sam3_ros',
        executable='sam3_node',
        name='sam3_node',
        output='screen',
        parameters=[
        {'target_object': target_object}
        ]
    )

    return LaunchDescription([
        realsense,
        target_object_arg,
        sam3_node
    ])