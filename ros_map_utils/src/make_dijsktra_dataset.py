#!/usr/bin/env python3
import rospy
import tf
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from navfn.srv import MakeNavPlan 
import numpy as np
import random
import pickle
import cv2 as cv
import time

class DjisktraCaller():
    def __init__(self):
        pass
        

        self.br = tf2_ros.TransformBroadcaster()

        self.poses_list = None
        
        self.poses_path = "/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120/paths/"        

        self.ranges = list(range(1012, 1100))
        self.load_poses(self.ranges)

        self.N_gen = 4
        self._idx_counter = 0

        self.x, self.y, self.yaw = 0, 0, 0


        map_msg = rospy.wait_for_message("/map", OccupancyGrid, timeout=2)
        self.map_resolution = map_msg.info.resolution
        self.map_origin = (map_msg.info.origin.position.x, map_msg.info.origin.position.y)
        self.map_size = (map_msg.info.height, map_msg.info.width)
        self.map = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.map = self.inflate_obstacles(self.map.astype(np.int8), 0.45, 0.05)  # Ensure it's in int8 format

        print(f"Map resolution: {self.map_resolution}")

        self.out_pkl_file = "/home/gonz/Desktop/THESIS/code/global-planning/gnd_dataset/local_map_files_120/paths/djisktra_paths.pkl"
        self.out_dict = {}

        self.timer = rospy.Timer(rospy.Duration(1 / 100), self.publish_tf_loop)

        rospy.wait_for_service('/navfn/make_plan', timeout=5)


    def load_poses(self, ranges):
        """
        Load poses from a given range.
        """
        rospy.loginfo(f"Loading poses...")

        self.poses_list = []
        for i in ranges:
            try:
                with open(f"{self.poses_path}/{i}_0.pkl", "rb") as f:
                    pose = pickle.load(f)
                    self.poses_list.append(pose['pose'])

            except FileNotFoundError:
                rospy.logwarn(f"Pose file for index {i} not found.")

        rospy.loginfo(f"Done loading {len(self.poses_list)} poses.")


    def inflate_obstacles(self, occupancy_map, robot_radius_m, resolution_m_per_px):
        radius_px = int(robot_radius_m / resolution_m_per_px)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radius_px+1, 2*radius_px+1))
        inflated = cv.dilate((occupancy_map > 0).astype(np.uint8), kernel)
        return inflated


    def world_to_map(self, x, y):
        x_map = int((x - self.map_origin[0]) / self.map_resolution)
        y_map = int((y - self.map_origin[1]) / self.map_resolution)

        return x_map, y_map
    
    def map_to_world(self, mx, my):
        wx = self.map_origin[0] + mx * self.map_resolution
        wy = self.map_origin[1] + my * self.map_resolution
        return wx, wy


    def is_valid_goal(self, x, y):
        mx, my = self.world_to_map(x, y)  # your conversion func
        if 0 <= mx < self.map_size[1] and 0 <= my < self.map_size[0]:
            return self.map[my, mx] == 0  # 0 = free
        return False
    
    def search_goals(self, K, R_min, R_max):
        t0 = time.time()
        
        # 1. Get all free map indices
        free_indices = np.argwhere(self.map == 0)  # shape (N, 2), [row, col]

        # 2. Convert to world coordinates (vectorized)
        mx = free_indices[:, 1]
        my = free_indices[:, 0]
        wx, wy = self.map_to_world(mx, my)  # should return arrays (N,), (N,)
        
        # 3. Vectorized geometric filtering
        dx = wx - self.x
        dy = wy - self.y
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx) - self.yaw
        theta = np.arctan2(np.sin(theta), np.cos(theta))  # normalize to [-π, π]

        mask = (r >= R_min) & (r <= R_max) & (np.abs(theta) <= np.deg2rad(100))

        valid_x = wx[mask]
        valid_y = wy[mask]
        
        # 4. Shuffle and choose up to K
        if len(valid_x) == 0:
            return []

        idxs = np.random.permutation(len(valid_x))[:K]
        goals = list(zip(valid_x[idxs], valid_y[idxs]))

        t1 = time.time()
        rospy.loginfo(f"Found {len(goals)} goals in {t1 - t0:.4f} seconds.")
        
        return goals



    def set_pose(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw + np.pi/2

    def select_pose(self):
        pass


    def publish_tf_loop(self, event):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = tf.transformations.quaternion_from_euler(0, 0, self.yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.br.sendTransform(t)


    def call_make_plan(self, start, goal):
        try:
            make_plan = rospy.ServiceProxy('/navfn/make_plan', MakeNavPlan)
            resp = make_plan(start, goal)
            rospy.loginfo(f"Got plan with {len(resp.path)} waypoints.")
            return resp
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return None
        

    def choose_goal(self):
        return random.choice(self.goals)


    @staticmethod
    def build_PoseStamped(x, y, yaw):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg
    

    @staticmethod
    def get_path_lenght_interval(odometry_xy, start, lenght=15., N_wpts=15):
        xsq = (odometry_xy[start+1:, 0] - odometry_xy[start:-1, 0]) ** 2
        ysq = (odometry_xy[start+1:, 1] - odometry_xy[start:-1, 1]) ** 2
        distances = np.cumsum(np.sqrt(xsq + ysq), axis=0)

        stop_idx = np.where(distances >= lenght)[0][0] + start
        ids = np.linspace(start, stop_idx, N_wpts, dtype=np.int64) # inlcude last point

        return odometry_xy[ids]


    def main(self):
        # rate = rospy.Rate()
        while self._idx_counter < len(self.poses_list) and not rospy.is_shutdown():
            # 1. choose a pose to call
            self.set_pose(*self.poses_list[self._idx_counter])

            # 2. make N calls to navfn
            start = self.build_PoseStamped(self.x, self.y, self.yaw)
            
            # search for goals
            goals = self.search_goals(K=self.N_gen, R_min=14, R_max=20)

            sampled_paths = []

            for goal in goals:
                xg, yg = goal 
                end   = self.build_PoseStamped(xg, yg, 0)

                plan_ret = self.call_make_plan(start, end)

                if plan_ret is not None and len(plan_ret.path) > 0:
                    plan_ret = np.array([[p.pose.position.x, p.pose.position.y] for p in plan_ret.path])
                    
                    try:
                        sampled_path = self.get_path_lenght_interval(plan_ret, 0, lenght=15., N_wpts=15)
                        sampled_paths.append(sampled_path)

                    except IndexError:
                        rospy.logwarn("Path not long enough. Skipping this goal.")
                        continue

                else:
                    continue

                # here you do something with the result if it is not None
                # like save it accordingly to the start pose you have and the local map and so on...
                # if plan_ret is not None:
                #     # do algo
                # rate.sleep()
                # rospy.sleep(0.1)
                self.out_dict[f"{self._idx_counter+self.ranges[0]}_{0}"] = sampled_paths

            self._idx_counter += 1
            
        # save the results
        # with open(self.out_pkl_file, "wb") as f:
        #     pickle.dump(self.out_dict, f)
        #     rospy.loginfo(f"Saved {len(self.out_dict)} paths to {self.out_pkl_file}")

        rospy.signal_shutdown("done")


if __name__=="__main__":

    rospy.init_node("bullshitter")
    
    dc = DjisktraCaller()

    rospy.sleep(1) # give time for the tf to start

    dc.main()
