#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FrameExtractorNode(Node):
    def __init__(self):
        super().__init__('frame_extractor')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(0.1, self.timer_callback)  # ~10 FPS

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub.publish(msg)

def main():
    rclpy.init()
    node = FrameExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()