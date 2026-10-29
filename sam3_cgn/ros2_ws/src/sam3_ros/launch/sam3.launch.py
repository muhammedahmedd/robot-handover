from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Allow the target object to be specified when launching
    target_object_arg = DeclareLaunchArgument(
        'target_object',
        default_value='bottle'
    )

    target_object = LaunchConfiguration('target_object')

    # Launch SAM3 segmentation node
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
        target_object_arg,
        sam3_node
    ])