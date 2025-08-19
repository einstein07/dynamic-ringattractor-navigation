from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        # Declare parameter file argument (default to parameters.json in package)
        DeclareLaunchArgument(
            'param_file',
            default_value=os.path.join(
                os.getenv('COLCON_PREFIX_PATH').split(':')[0],
                'share', 'ringattractor', 'parameters.json'
            ),
            description='Full path to the parameters.json file'
        ),

        # Start control.py node
        Node(
            package='ringattractor',
            executable='ringattractor',
            name='ringattractor',
            output='screen',
            arguments=[LaunchConfiguration('param_file')]
        )
    ])
