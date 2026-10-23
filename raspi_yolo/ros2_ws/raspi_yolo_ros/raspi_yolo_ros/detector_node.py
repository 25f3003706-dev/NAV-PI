#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import torch
from models.raspi_yolo import RaspiYOLO
# ... import your decode + NMS utils

class DetectorNode(Node):
    def __init__(self):
        super().__init__('raspi_detector')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.bridge = CvBridge()
        self.model = RaspiYOLO(...)
        self.model.load_state_dict(torch.load(...))
        self.model.eval()
        self.model.reset_memory()

    def callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # preprocess → tensor
        with torch.no_grad():
            det_outs, temporal, _ = self.model(tensor, mode='temporal')
        # decode → Detection2DArray
        det_msg = Detection2DArray()
        det_msg.header = msg.header
        # fill detections...
        self.pub.publish(det_msg)

def main():
    rclpy.init()
    node = DetectorNode()
    rclpy.spin(node)
    ...