#!/usr/bin/env python
"""
A ROS node that uses the detection and tracking capabilities of this package
to control vehicle steering to follow people.

Author:
    Alex Tyshka -- atyshka@dataspeedinc.com
"""

# ROS
import rospy

import cv2

import threading

from ds_object_detection.msg import Detection, DetectionArray, Rect

from sensor_msgs.msg import Image

from dbw_gem_msgs.msg import SteeringCmd

from cv_bridge import CvBridge, CvBridgeError

from ds_object_detection_lib import utils

from object_detection.utils import visualization_utils as vis_util

class PeopleObjectDetectionNode(object):
    def __init__(self):
        super(PeopleObjectDetectionNode, self).__init__()

        # init the node
        rospy.init_node('people_object_detection', anonymous=False)

        self._bridge = CvBridge()

        self.pub_detections_image = rospy.Publisher(\
            '/person_tracking/tracking_image', Image, queue_size=1)
            
        self.pub_steering_cmd = rospy.Publisher(\
            '/vehicle/steering_cmd', SteeringCmd, queue_size=1)

        self.sub_detections = rospy.Subscriber('/object_tracker/tracks', \
            DetectionArray, self.detection_callback, queue_size=1)

        self.sub_rgb = rospy.Subscriber('/camera/color/image_raw',\
            Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        self.cached_image = None
        
        self.track_id = 0
        
        self.reset_threshold = 25
        
        self.frames_missing = 50
        
        self.steering_cmd = SteeringCmd()
        self.steering_cmd.steering_wheel_angle_cmd = 0.0
        self.steering_cmd.enable = True
        self.steering_cmd.ignore = True
        
        rospy.Timer(rospy.Duration(0.02), self.pub_cmd)
        
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

    def pub_cmd(self, event):
        self.pub_steering_cmd.publish(self.steering_cmd)
        
    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("Shutting down!")

    def rgb_callback(self, data):
        """
        Callback for RGB images from the camera
        """
        self.cached_image = data
           
    def detection_callback(self, msg):
        """
        Callback for bounding box detections from the neural net
        """
        object_ids = []
        rois = []
        for detection in msg.detections:
            if detection.label == 'person':
                object_ids.append(detection.id)
                rois.append(detection.mask.roi)
        
        if len(object_ids) < 1:
            self.frames_missing += 1
            self.pub_detections_image.publish(self.cached_image)
            return
        
        if self.track_id in object_ids:
            #Found object, reset counter and draw a bounding box
            self.frames_missing = 0
            roi = rois[object_ids.index(self.track_id)]
            #Number between -1 and 1 for object position
            x_relative = ((roi.x + roi.x + roi.width) / float(self.cached_image.width)) - 1.0
            
            self.steering_cmd.steering_wheel_angle_cmd = x_relative * -10
            
            try:
                cv_image = self._bridge.imgmsg_to_cv2(self.cached_image, "bgr8")
                vis_util.draw_bounding_box_on_image_array(
                    cv_image,
                    roi.y,
                    roi.x,
                    roi.y + roi.height,
                    roi.x + roi.width,
                    use_normalized_coordinates=False)
                msg_im = self._bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
                self.pub_detections_image.publish(msg_im)
            except CvBridgeError as e:
                print(e)
        else:
            self.frames_missing += 1
            if self.frames_missing > self.reset_threshold:
                self.track_id = min(object_ids)
            self.pub_detections_image.publish(self.cached_image)
                

def main():
    node = PeopleObjectDetectionNode()

if __name__ == '__main__':
    main()
