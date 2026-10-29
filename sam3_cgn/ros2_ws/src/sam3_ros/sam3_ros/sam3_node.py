import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import torch

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from perception_interfaces.msg import TargetSegmentation

from PIL import Image as PILImage
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Node(Node):

    def __init__(self):
        super().__init__('sam3_node')

        # Declare the target object ROS parameter and store its current value
        self.declare_parameter('target_object', 'bottle')
        self.target_object = self.get_parameter('target_object').value

        # Load the model
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

        self.get_logger().info('SAM3 loaded')

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.segmentation_pub = self.create_publisher(
            TargetSegmentation,
            '/sam3/target_seg',
            1
        )

        self.get_logger().info('SAM3 ROS node started')

    def image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='rgb8'
        )
       
        image = PILImage.fromarray(cv_image)   

        # Prevent data type mismatch during SAM3 inference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

            # Load an image  
            state = self.processor.set_image(image)

            output = self.processor.set_text_prompt(
                state=state,
                prompt=self.target_object
            )

        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

        self.get_logger().info(
            f'SAM3 found {len(masks)} object(s)'
        )

        if len(masks) > 0:
            best_idx = torch.argmax(scores)
            mask = masks[best_idx]

            mask = mask.squeeze().cpu().numpy()
            mask = (mask * 255).astype('uint8')

            mask_msg = self.bridge.cv2_to_imgmsg(
                mask,
                encoding='mono8'
            )

            mask_msg.header = msg.header

            seg_msg = TargetSegmentation()
            seg_msg.rgb = msg
            seg_msg.mask = mask_msg

            self.segmentation_pub.publish(seg_msg)



def main(args=None):
    rclpy.init(args=args)

    node = SAM3Node()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()