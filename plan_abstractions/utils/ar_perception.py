from perception_utils.apriltags import AprilTagDetector
import os
import time
from perception_utils.box_detector import BoxDetector, InHandBoxDetector
from perception_utils.realsense import get_first_realsense_sensor
from perception import Kinect2SensorFactory, KinectSensorBridged
from sensor_msgs.msg import Image
import rospy
from perception.camera_intrinsics import CameraIntrinsics
from autolab_core import RigidTransform, YamlConfig
import numpy as np
from pyquaternion import Quaternion

def quaternion_dist(q1, q2):
    return Quaternion.absolute_distance(Quaternion(q1), Quaternion(q2))
print("waiting for camera")

class ObjectDetector:
    def __init__(self, cfg):
        self.cfg = cfg # YamlConfig("data/calibration/april_tag_pick_place_azure_kinect_cfg.yaml")
        self._logdir = os.getcwd()
        num_tags_per_rod = 4
        self._detection_counter = 0
        dist_to_mid_tag = 0.02
        dist_to_far_tag = 0.068 #make sure matches with stickers
        from_fr = "rod"
        to_fr = "tag"
        self._rod_to_tags = [RigidTransform(translation=np.array([-dist_to_far_tag,0,0]), from_frame=from_fr, to_frame=to_fr),
                             RigidTransform(translation=np.array([-dist_to_mid_tag,0,0]), from_frame=from_fr, to_frame=to_fr),
                             RigidTransform(translation=np.array([dist_to_mid_tag,0,0]), from_frame=from_fr, to_frame=to_fr),
                             RigidTransform(translation=np.array([dist_to_far_tag,0,0]), from_frame=from_fr, to_frame=to_fr),
                             ]
        self.T_camera_world = RigidTransform.load(self.cfg['T_k4a_franka_path'])
        self.sensor = Kinect2SensorFactory.sensor('bridged', self.cfg)  # Kinect sensor object
        prefix = self.cfg["prefix"]
        self.sensor.topic_image_color = prefix + self.sensor.topic_image_color
        self.sensor.topic_image_depth = prefix + self.sensor.topic_image_depth
        self.sensor.topic_info_camera = prefix + self.sensor.topic_info_camera
        self.sensor.start()
        self._is_april = False
        if self._is_april:
            self.detector = AprilTagDetector(self.cfg['april_tag'])
        else:
            self.detector = BoxDetector(self.cfg['color_segment'])
        # intr = sensor.color_intrinsics #original
        if "overhead" in cfg["prefix"]:
            #overhead_intr = CameraIntrinsics('k4a', fx=970.4990844726562, cx=1025.4967041015625, fy=970.1990966796875,cy=777.769775390625, height=1536, width=2048)  # fx fy cx cy overhead
            overhead_intr = CameraIntrinsics('k4a', fx=979.04345703125, cx=1019.5509033203125, fy=978.8479614257812,cy=777.769775390625, height=1536, width=2048)  # fx fy cx cy overhead
            self.intr = overhead_intr
        else:
            side_intr = CameraIntrinsics('k4a', fx=978.7222900390625, cx=1019.9566650390625, fy=978.6996459960938,cy=782.781982421875, height=1536, width=2048)  # fx fy cx cy overhead
            self.intr = side_intr

        #overhead_intr = CameraIntrinsics('k4a', fx=1819.685791015625, cx=1923.2437744140625, fy=1819.123291015625,cy=1098.7557373046875, height=1536, width=2048)  # fx fy cx cy overhead

    def detect(self, num_tries = 3, debug=False):
        T_tag_cameras = []
        for _ in range(num_tries):
            logdir_for_detection = os.path.join(self._logdir, f"{self._detection_counter}")
            if not os.path.isdir(logdir_for_detection):
                os.mkdir(logdir_for_detection)
            detections = self.detector.detect(self.sensor, self.intr, self.cfg['vis_detect'], logdir=logdir_for_detection)
            if len(detections) > 0:
                break
            print("Trying again")
            time.sleep(0.8)
        if len(detections) == 0:
            if debug:
                print("Still no detections. Needs debugging")
                detections = self.detector.detect(self.sensor, self.intr, self.cfg['vis_detect'])
            else:
                return [], []
        detected_ids = []
        for new_detection in detections:
            detected_ids.append(int(new_detection.from_frame.split("/")[1])) #won't work for non-int values
            T_tag_cameras.append(new_detection)
        T_tag_worlds = []
        for T_tag_camera in T_tag_cameras:
            T_tag_camera.to_frame="kinect2_overhead"
            T_tag_world = self.T_camera_world * T_tag_camera
            T_tag_worlds.append(T_tag_world)
        self._detection_counter +=1
        return detected_ids, T_tag_worlds


    def get_rod_rt_from_detections(self, rod_num, detections):
        if self._is_april:
            return self._get_rod_rt_from_april_detections(rod_num, detections)
        detected_ids, T_tag_worlds = detections
        detected_ids = np.array(detected_ids)
        if rod_num not in detected_ids:
            #raise RuntimeError("Rod not detected")
            print("Rod not detected")
            return None
        relevant_ids =  np.where(detected_ids == rod_num)[0]
        relevant_rts = np.array(T_tag_worlds)[relevant_ids] #should be two
        assert (len(relevant_rts) == 2)
        endpoint_1_rt = [tag for tag in relevant_rts if tag.from_frame.split("_")[2] == '0'][0]
        endpoint_2_rt = [tag for tag in relevant_rts if tag.from_frame.split("_")[2] == '1'][0]
        del_y = endpoint_2_rt.translation[1] - endpoint_1_rt.translation[1]
        del_x = endpoint_2_rt.translation[0] - endpoint_1_rt.translation[0]
        midpoint = 0.5*(endpoint_1_rt.translation + endpoint_2_rt.translation)
        #z tends to be sketchy for some reason
        if abs(endpoint_1_rt.translation[2] - endpoint_2_rt.translation[2]) > 0.4: #some large number
            print("One midpoint z is very different from the other. Normally this happens with a gap in the depth camera")
            midpoint[2] = min(endpoint_1_rt.translation[2], endpoint_2_rt.translation[2])
        
        yaw = np.arctan2(del_y, del_x) - np.pi/2
        rotation = RigidTransform.z_axis_rotation(yaw)
        rt = RigidTransform(translation = midpoint, rotation = rotation, from_frame="tag", to_frame="world")
        return rt


    def _average_curr_estimate(self, curr_estimates):
        print(f"Estimating from {len(curr_estimates)} RTs")
        quats = [curr_estimate.quaternion for curr_estimate in curr_estimates]
        poses = [curr_estimate.translation for curr_estimate in curr_estimates]
        average_translation = np.mean(poses, axis=0)
        average_quaternion = np.mean(quats, axis=0) #might need to account for double problem
        ref_rt = curr_estimates[0]
        ref_rt.translation = average_translation
        ref_rt.rotation = RigidTransform.rotation_from_quaternion(average_quaternion)
        return ref_rt

    def _get_rod_rt_from_april_detections(self, rod_num, detections):
        detected_ids, T_tag_worlds = detections
        if rod_num == 0:
            rel_ids = [0,1,2,3]
        else:
            rel_ids = [0,1,2,3]

        self._rod_to_tags
        curr_estimates = []
        for id, T_tag_world in zip(rel_ids, T_tag_worlds):
            T_tag_world.from_frame = "tag"
            if id in detected_ids:
                relevant_rod_to_tag = self._rod_to_tags[rel_ids.index(id)]
                curr_estimate = T_tag_world * relevant_rod_to_tag
                curr_estimates.append(curr_estimate)
        #Average, potentially remove outliers
        return self._average_curr_estimate(curr_estimates)

class InHandRodDetector:
    def __init__(self, cfg):
        self.cfg = cfg # YamlConfig("data/calibration/april_tag_pick_place_azure_kinect_cfg.yaml")
        from_fr = "rod"
        to_fr = "tag"
        self._detection_counter=0
        self._logdir=None
        self.T_camera_world = RigidTransform()
        self.sensor = get_first_realsense_sensor(cfg['rs']) 
        self.sensor.start()
        self.detector = InHandBoxDetector(self.cfg['color_segment'])
        # intr = sensor.color_intrinsics #original
        overhead_intr = CameraIntrinsics('k4a', fx=616.75732421875, cx=319.7129821777344, fy=616.644775390625,cy=246.04098510742188, height=480, width=640)  # fx fy cx cy overhead
        self.intr = overhead_intr

    def detect(self, num_tries = 3):
        T_tag_cameras = []
        for _ in range(num_tries):
            detections = self.detector.detect(self.sensor, self.intr, self.cfg['vis_detect'], logdir=os.path.join(self._logdir, f"/{self._detection_counter}"))
            if len(detections) > 0:
                break
            print("Trying again")
            time.sleep(0.8)
        if len(detections) == 0:
            import ipdb; ipdb.set_trace()
            print("Still no detections. Needs debugging")
            detections = self.detector.detect(self.sensor, self.intr, 1)
            print("By continuing, the assumption is that the rod that is occluded hasn't moved")
            detections = [0] 
        self._detection_counter +=1
        return detections




if __name__ == "__main__":
    rospy.init_node("perception")
    cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/rod_detect.yaml")
    #cfg = YamlConfig("/home/lagrassa/git/plan-abstractions/cfg/perception/drawer_detect.yaml")
    detector = ObjectDetector(cfg)
    for i in range(2):
        detections = detector.detect()
        rod_num = i
        rt = detector.get_rod_rt_from_detections(rod_num, detections)
        print(rt.translation)

