#!/usr/bin/python
#-*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf

import rospkg
import sys

from scipy.interpolate import interp1d

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Point
from object_msgs.msg import Object
from visualization_msgs.msg import Marker

import matplotlib.pyplot as plt

import time
import pickle
import argparse
import stanley
import optimal_trajectory_Frenet as opt_traj



rospack = rospkg.RosPack()
path = rospack.get_path("map_server")

rn_id = dict()

rn_id[5] = {
    'left': [18, 2, 11, 6, 13, 8, 15, 10, 26, 0]  # ego route
}

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def interpolate_waypoints(wx, wy, space=0.5):
    _s = 0
    s = [0]
    for i in range(1, len(wx)):
        prev_x = wx[i - 1]
        prev_y = wy[i - 1]
        x = wx[i]
        y = wy[i]

        dx = x - prev_x
        dy = y - prev_y

        _s = np.hypot(dx, dy)
        s.append(s[-1] + _s)

    fx = interp1d(s, wx)
    fy = interp1d(s, wy)
    ss = np.linspace(0, s[-1], num=int(s[-1] / space) + 1, endpoint=True)

    dxds = np.gradient(fx(ss), ss, edge_order=1)
    dyds = np.gradient(fy(ss), ss, edge_order=1)
    wyaw = np.arctan2(dyds, dxds)

    return {
        "x": fx(ss),
        "y": fy(ss),
        "yaw": wyaw,
        "s": ss
    }


class State:
    
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1, WB=2.6):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.dt = dt
        self.WB = WB
        self.max_steering = np.radians(30)

    def update(self, a, delta):
        dt = self.dt
        WB = self.WB

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.yaw = pi_2_pi(self.yaw)
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

def obj_cb2(data):
    global obs2
    obs2 = [data.x, data.y, data.yaw]

def obj_cb3(data):
    global obs3
    obs3 = [data.x, data.y, data.yaw]

def get_ros_msg(x, y, yaw, v, id):
    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

    m = Marker()
    m.header.frame_id = "/map"
    m.header.stamp = rospy.Time.now()
    m.id = id
    m.type = m.CUBE

    m.pose.position.x = x + 1.3 * math.cos(yaw)
    m.pose.position.y = y + 1.3 * math.sin(yaw)
    m.pose.position.z = 0.75
    m.pose.orientation = Quaternion(*quat)

    m.scale.x = 4.475
    m.scale.y = 1.850
    m.scale.z = 1.645

    m.color.r = 93 / 255.0
    m.color.g = 122 / 255.0
    m.color.b = 177 / 255.0
    m.color.a = 0.97

    o = Object()
    o.header.frame_id = "/map"
    o.header.stamp = rospy.Time.now()
    o.id = id
    o.classification = o.CLASSIFICATION_CAR
    o.x = x
    o.y = y
    o.yaw = yaw
    o.v = v
    o.L = m.scale.x
    o.W = m.scale.y

    return {
        "object_msg": o,
        "marker_msg": m,
        "quaternion": quat
    }

def visualizaiton_trajectory(fplist, opt_ind, mpax, mapy, maps):
    global marker_traj_pub, marker_traj
    
    temp2 = []
    point = Point()
    print(type(fplist), len(fplist), opt_ind)
    for i in range(len(fplist[opt_ind].s)):
        x, y, heading = opt_traj.get_cartesian(fplist[opt_ind].s[i], fplist[opt_ind].d[i], mpax, mapy, maps)
        point.x = x
        point.y = y
        temp2.append(point)
    #print(temp2)
    marker_traj.points = temp2
    marker_traj_pub.publish(marker_traj)
    #print(fplist[opt_ind].s, fplist[opt_ind].d)

    #temp2 = []
    #for fp in fplist:


if __name__ == "__main__":
    global obs2, obs3
    obs2 = None
    obs3 = None
    parser = argparse.ArgumentParser(description='Spawn a CV agent')

    parser.add_argument("--id", "-i", type=int, help="agent id", default=1)
    parser.add_argument("--route", "-r", type=int,
                        help="start index in road network. select in [1, 3, 5, 10]", default=5)
    parser.add_argument("--dir", "-d", type=str, default="left", help="direction to go: [left, straight, right]")
    args = parser.parse_args()

    rospy.init_node("three_cv_agents_node_" + str(args.id))

    # test Marker
    marker_traj = Marker()
    marker_traj.header.frame_id = "/map"
    marker_traj.header.stamp = rospy.get_rostime()
    marker_traj.ns = "ego_car"
    marker_traj.id = 0
    marker_traj.action = Marker.ADD
    
    marker_traj.scale.x = 100.0
    marker_traj.scale.y = 100.0
    marker_traj.scale.z = 100.0

    marker_traj.color.r = 1.0
    marker_traj.color.g = 0.0
    marker_traj.color.b = 0.0

    marker_traj.color.a = 1.0
    marker_traj.lifetime = rospy.Duration(0.5)

    marker_traj.type = Marker.SPHERE_LIST


    id = args.id
    tf_broadcaster = tf.TransformBroadcaster()
    topic = 'visualization_marker'
    marker_traj_pub = rospy.Publisher(topic, Marker, queue_size=1)
    marker_pub = rospy.Publisher("/objects/marker/car_" + str(id), Marker, queue_size=1)
    object_pub = rospy.Publisher("/objects/car_" + str(id), Object, queue_size=1)
    rospy.Subscriber("/objects/car_2", Object, obj_cb2)
    rospy.Subscriber("/objects/car_3", Object, obj_cb3)
    
    
    start_node_id = args.route
    route_id_list = [start_node_id] + rn_id[start_node_id][args.dir]

    ind = 100

    with open(path + "/src/route.pkl", "rb") as f:
        nodes = pickle.load(f)

    wx = []
    wy = []
    wyaw = []
    for _id in route_id_list:
        wx.append(nodes[_id]["x"][1:])
        wy.append(nodes[_id]["y"][1:])
        wyaw.append(nodes[_id]["yaw"][1:])
    wx = np.concatenate(wx)
    wy = np.concatenate(wy)
    wyaw = np.concatenate(wyaw)

    waypoints = {"x": wx, "y": wy, "yaw": wyaw}

    target_speed = 20.0 / 3.6
    state = State(x=waypoints["x"][ind], y=waypoints["y"][ind], yaw=waypoints["yaw"][ind], v=5.0, dt=0.01)

    r = rospy.Rate(100)
    a = 0

    

    # get map
    mapx = waypoints["x"]
    mapy = waypoints["y"]
    maps = np.zeros(mapx.shape)
    for i in range(len(mapx)):
        x = mapx[i]
        y = mapy[i]
        sd = opt_traj.get_frenet(state.x, state.y, mapx, mapy)
        maps[i] = sd[0]
    
    time.sleep(2)
    ## temo obstacle
    temp2 = opt_traj.get_frenet(obs2[0], obs2[1], mapx, mapy)
    temp3 = opt_traj.get_frenet(obs3[0], obs3[1], mapx, mapy)
    WIDTH = 0.19  # [m]
    obs = np.array([[temp2[0], temp2[1]*WIDTH],
                    [temp3[0], temp3[1]*WIDTH]])
    print (obs)

    while not rospy.is_shutdown():
        # generate acceleration ai, and steering di
        # YOUR CODE HERE
        ## update condition

        s, d = opt_traj.get_frenet(state.x, state.y, mapx, mapy)
        print(s, d)
        x, y, yaw_road = opt_traj.get_cartesian(s, d, mapx, mapy, maps)
        yawi = state.yaw - yaw_road

        # s 방향 초기조건
        si = s
        si_d = state.v*np.cos(yawi)
        si_dd = a*np.cos(yawi)
        sf_d = target_speed
        sf_dd = 0

        # d 방향 초기조건
        di = d
        di_d = state.v*np.sin(yawi)
        di_dd = a*np.sin(yawi)
        df_d = 0
        df_dd = 0

        opt_d = di

        
        path, opt_ind = opt_traj.frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, mapx, mapy, maps, opt_d)
        #visualizaiton_trajectory(path, opt_ind, mapx, mapy, maps)

        # consistency cost를 위해 update
        opt_d = path[opt_ind].d[-1]
        
        traj_x = []
        traj_y = []
        traj_yaw = []
        print('---traj---')
        for i in range(len(path[opt_ind].s)):
            temp_x, temp_y, temp_yaw = opt_traj.get_cartesian(path[opt_ind].s[i], path[opt_ind].d[i], mapx, mapy, maps)
            #print(temp_x, temp_y, temp_yaw)
            traj_x.append(temp_x)
            traj_y.append(temp_y)
            traj_yaw.append(temp_yaw)
        
        #steer = stanley.stanley_control(state.rear_x, state.rear_y, state.yaw, state.v, mapx, mapy, waypoints["yaw"])
        steer = stanley.stanley_control(state.rear_x, state.rear_y, state.yaw, state.v, traj_x, traj_y, traj_yaw)
        d = np.clip(steer, -state.max_steering, state.max_steering)
        print(state.x, state.y, state.yaw, steer)
        # update state with acc, delta
        state.update(a, d)

        # si = path[opt_ind].s[1]
        # si_d = path[opt_ind].s_d[1]
        # si_dd = path[opt_ind].s_dd[1]
        # di = path[opt_ind].d[1]
        # di_d = path[opt_ind].d_d[1]
        # di_dd = path[opt_ind].d_dd[1]
        
        # vehicle state --> topic msg
        msg = get_ros_msg(state.x, state.y, state.yaw, state.v, id=id)

        # send tf
        tf_broadcaster.sendTransform(
            (state.x, state.y, 1.5),
            msg["quaternion"],
            rospy.Time.now(),
            "/car_" + str(id), "/map"
        )

        # publish vehicle state in ros msg
        object_pub.publish(msg["object_msg"])

        r.sleep()
    
    ## dibug
    plt.figure()
    plt.plot(np.array(range(len(mapx))), mapx)
    print(type(mapx), np.shape(mapx))
    plt.savefig('mapx')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(range(len(path[opt_ind].s)), path[opt_ind].s)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(range(len(path[opt_ind].d)), path[opt_ind].d)
    plt.savefig('path_sd')

    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    ax1.plot(range(len(traj_x)), traj_x)
    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(range(len(traj_y)), traj_y)
    ax3 = fig.add_subplot(1,3,3)
    ax3.plot(range(len(traj_yaw)), traj_yaw)
    plt.savefig('traj_x,y,yaw')