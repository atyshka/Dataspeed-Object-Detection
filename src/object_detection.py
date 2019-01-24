#!/usr/bin/env python
"""
A ROS node to detect objects via TensorFlow Object Detection API.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

# ROS
import rospy

import cv2

from copy import deepcopy

from ds_object_detection.msg import Detection, DetectionArray, Rect

from sensor_msgs.msg import Image, PointCloud2

from cv_bridge import CvBridge, CvBridgeError

from ds_object_detection_lib.detector import Detector

from ds_object_detection_lib import utils

class PeopleObjectDetectionNode(object):
    """docstring for PeopleObjectDetectionNode."""
    def __init__(self):
        super(PeopleObjectDetectionNode, self).__init__()

        # init the node
        rospy.init_node('people_object_detection', anonymous=False)

        # Get the parameters
        (model_name, num_of_classes, label_file, camera_namespace, video_name,
            num_workers) \
        = self.get_parameters()

        # Create Detector
        self._detector = Detector(model_name, num_of_classes, label_file,
            num_workers)

        self._bridge = CvBridge()

        self._cached_cloud = PointCloud2()

        # Advertise the result of Object Detector
        self.pub_detections = rospy.Publisher('/object_detection/detections', \
            DetectionArray, queue_size=1)

        # Advertise the result of Object Detector
        self.pub_detections_image = rospy.Publisher(\
            '/object_detection/detections_image', Image, queue_size=1)

        # Advertise synced cloud
        self.pub_cloud = rospy.Publisher(\
            '/object_detection/cloud_synced', PointCloud2, queue_size=1)

        # Subscribe to the face positions
        self.sub_rgb = rospy.Subscriber(camera_namespace,\
            Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        # This subscriber is for syncing the cloud with the detection
        self.sub_cloud = rospy.Subscriber('/camera/depth/color/points', \
            PointCloud2, self.cloud_callback, queue_size=1)
        
        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (model name, num_of_classes, label_file)

        """

        model_name  = rospy.get_param("~model_name")
        num_of_classes  = rospy.get_param("~num_of_classes")
        label_file  = rospy.get_param("~label_file")
        camera_namespace  = rospy.get_param("~camera_namespace")
        video_name = rospy.get_param("~video_name")
        num_workers = rospy.get_param("~num_workers")

        return (model_name, num_of_classes, label_file, \
                camera_namespace, video_name, num_workers)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def cloud_callback(self, data):
        self._cached_cloud = data

    def rgb_callback(self, data):
        """
        Callback for RGB images
        """
        # Copy this cloud to uniquely associate it with the image: 
        # we don't want the cloud changing on us while detecting
        cloud_copy = deepcopy(self._cached_cloud)
        try:
            # Convert image to numpy array
            cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")

            # Detect
            (output_dict, category_index) = \
                self._detector.detect(cv_image)


            # Create the message
            msg = \
                utils.create_detection_msg(\
                data, output_dict, category_index, self._bridge)

            # Draw bounding boxes
            image_processed = \
                self._detector.visualize(cv_image, output_dict)

            # Convert numpy image into sensor img
            msg_im = \
                self._bridge.cv2_to_imgmsg(\
                image_processed, encoding="passthrough")

            # Publish the messages
            self.pub_cloud.publish(cloud_copy)
            self.pub_detections.publish(msg)
            self.pub_detections_image.publish(msg_im)

        except CvBridgeError as e:
            print(e)

def main():
    """ main function
    """
    node = PeopleObjectDetectionNode()

if __name__ == '__main__':
    main()
