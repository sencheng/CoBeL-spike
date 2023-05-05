"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import json
import os


class OpenFieldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30}

    def __init__(self, goal_velocity=0):
        self._read_json_p()
        self.min_position = np.array([self.xmin_position, self.xmax_position])
        self.max_speed = 0.07
        self.goal_position = np.array([self.goal_x, self.goal_y])
        # self.goal_velocity = goal_velocity

        # self.force=0.001
        # self.gravity=0.0025

        self.low = np.array([self.xmin_position, self.ymin_position])
        self.high = np.array([self.xmax_position, self.ymax_position])

        self.viewer = None

        self.action_space = spaces.Box(low=np.array([-10.0, -10.0, 0.0]), high=np.array([10.0, 10.0, 10000.0]),
                                       dtype=np.float32)  # Should read from a corresponding parameter file
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.last_action = 0.0
        self.action = 0.0

        #        self.reward_recep_field = 0.1

        self.seed()

        self.trial_num = 1
        self.cnt = 0
        self.cnt_begin = 0
        self.done = False
        self.cnt_rew_deliver = 5
        self.prev_tr_dur = 0.0

        self.f_tr_loc = open(os.path.join(self.data_path, 'locs_time.dat'), 'w')
        self.f_tr_loc.write('trial\ttime\tx\ty\n')

        self.f_tr_time_rew = open(os.path.join(self.data_path, 'trial_time_rew.dat'), 'w')
        self.f_tr_time_rew.write('trial\ttime\treward\n')
        
        


    def seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if len(action) == 1:
            # print('openfield-env: ', 'action vector has only one element. Neutral action is chosen!')
            action = np.array([0., 0.])
            nest_running = False
            runtime = 0.0
        else:
            runtime = action[-1]
            action = action[0:-1]
            nest_running = True
            self.cnt_begin += 1

        velocity = np.array(action)

        self.action = np.arctan2(action[0], action[1])
        position = self.state

        tmp_pos = position + velocity
        
        candidates = []
        # Check is the line segment pos->tmp_pos doesn't cross any line segment defined by the obstacle
        # TODO: if performance suffers, do a broad pass here
        for obstacle in self.obstacle_list:
            for i, point in enumerate(obstacle):
                obs_segment = (obstacle[i], obstacle[i - 1]) # Points need to be listed sequentially!
                if self._intersect(obs_segment[0], obs_segment[1], position, tmp_pos):
                    candidates.append(self._calc_collision_point(obs_segment[0], obs_segment[1], position, tmp_pos))
        
        #This handles cases where multiple segments are crossed
        if len(candidates) > 1:
            best = candidates[0]
            for candidate in candidates:
                if math.dist(tmp_pos, candidate) < math.dist(tmp_pos, best):
                    best = candidate
            tmp_pos = best
        elif len(candidates) == 1:
            tmp_pos = candidates[0]

        # Can maybe use numpy.clip here?
        if tmp_pos[0] > self.xmax_position:
            tmp_pos[0] = self.xmax_position
        elif tmp_pos[0] < self.xmin_position:
            tmp_pos[0] = self.xmin_position
        elif tmp_pos[1] > self.ymax_position:
            tmp_pos[1] = self.ymax_position
        elif tmp_pos[1] < self.ymin_position:
            tmp_pos[1] = self.ymin_position

        position = tmp_pos

        #field_geom_mean = np.sqrt((self.xmax_position - self.xmin_position) * (self.ymax_position - self.ymin_position))
        field_geom_mean = 1
        rew_territory = field_geom_mean * self.reward_recep_field
        done = bool(np.linalg.norm(position - self.goal_position) <= rew_territory)
        if self.hide_goal == True:
            done = False

        reward = -1

        self.f_tr_loc.write('{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(self.trial_num, runtime, position[0], position[1]))

        if done:
            reward = 1
            self.f_tr_time_rew.write('{:d}\t{:.5f}\t{:.1f}\n'.format(self.trial_num, runtime, reward))
            self.trial_num += 1
            self.prev_tr_dur = runtime
        elif runtime - self.prev_tr_dur >= self.max_tr_dur:
            self.f_tr_time_rew.write('{:d}\t{:.5f}\t{:.1f}\n'.format(self.trial_num, runtime, reward))
            self.prev_tr_dur = runtime
            self.trial_num += 1
            done = True

        self.state = position
        self.cnt += 1

        return np.array(self.state), reward, done, {}
    
    '''
    Calculates the intersection point between two line segments, so that corrections
    can be made if the agent attempts to move into an obstacle
    '''
    @staticmethod
    def _calc_collision_point(obs1, obs2, traj1, traj2):
        # TODO: Should this be moved to a parameter file?
        OFFSET = .05 # Offset value which defines how far to displace the agent from the obstacle border in the case of a collision.
        
        # Get vectors for obstacle line segment and trajectory
        obs_seg = np.subtract(obs2, obs1)
        traj_seg = np.subtract(traj2, traj1)
        
        # Use 2d cross products to find distance along the line at which lines intersect
        # TODO: it's theoretically possible to have some cases where determinant ~= 0
        dist1 = (obs_seg[0] * (obs1[1] - traj1[1]) - obs_seg[1] * (obs1[0] - traj1[0])) / (traj_seg[1] * obs_seg[0] - traj_seg[0] * obs_seg[1])
        dist2 = (traj_seg[0] * (obs1[1] - traj1[1]) - traj_seg[1] * (obs1[0] - traj1[0])) / (traj_seg[1] * obs_seg[0] - traj_seg[0] * obs_seg[1])
        
        if (dist1 >= 0 and dist1 <= 1 and dist2 >= 0 and dist2 <= 1):
            dist1 = max(dist1 - OFFSET, 0)
            x_int = traj1[0] + (dist1 * traj_seg[0])
            y_int = traj1[1] + (dist1 * traj_seg[1])
            return [x_int, y_int]
        else:
            # If this method is called at all, we expect a collision. This branch indicates some sort of bug, since we haven't found one!
            raise NotImplementedError
    
    
    #Checks if three given points are arranged in a counter-clockwise order. This is used for collision detection.
    #https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    @staticmethod
    def _ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    
    @staticmethod
    def _intersect(A, B, C, D):
        return OpenFieldEnv._ccw(A, C, D) != OpenFieldEnv._ccw(B, C, D) and OpenFieldEnv._ccw(A, B, C) != OpenFieldEnv._ccw(A, B, D)

    def reset(self):
        self.state = np.array([self.start_x, self.start_y])
        # testing border/obstacle -> action connections
#        shift = 0.6
#        if self.trial_num <= 2:
#            self.state = np.array([0, 1.05]) # Boundary ymax
#        elif self.trial_num <= 4:
#            self.state = np.array([1.05, 0]) # Boundary xmax
#        elif self.trial_num <= 6:
#            self.state = np.array([0, -1.05]) # Boundary ymin
#        elif self.trial_num <= 8:
#            self.state = np.array([-1.05, 0]) # Boundary xmin
#        elif self.trial_num <= 10:
#            self.state = np.array([0, 0.75]) # Obstacle ymax
#        elif self.trial_num <= 12:
#            self.state = np.array([0.25, 0]) # Obstacle xmax
#        elif self.trial_num <= 14:
#            self.state = np.array([0, -0.75]) # Obstacle ymin
#        elif self.trial_num <= 16:
#            self.state = np.array([-0.25, 0]) # Obstacle xmin
        
        # Testing path acquisition with obstacle
#        shift = 0.6
#        if self.trial_num <= 5:
#            self.state = np.array([self.start_x, self.start_y]) # (0.6, 0)
#        elif self.trial_num <= 10:
#            self.state = np.array([self.start_x, self.start_y - shift])
#        elif self.trial_num <= 15:
#            self.state = np.array([self.start_x - shift, self.start_y - shift])
#        elif self.trial_num <= 20:
#            self.state = np.array([self.start_x - (shift*2), self.start_y - shift])
#        elif self.trial_num <= 25:
#            self.state = np.array([self.start_x - (shift*2), self.start_y])
#        elif self.trial_num <= 30:
#            self.state = np.array([self.start_x - (shift*2), self.start_y + shift])
        
        return np.array(self.state)

    def _height(self, xs):
        return self.ymax_position - self.ymin_position

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.xmax_position - self.xmin_position
        world_height = self.ymax_position - self.ymin_position
        scale_w = screen_width / world_width
        scale_h = screen_height / world_height
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # xs = np.linspace(self.xmin_position, self.xmax_position, 100)
            # ys = np.linspace(self.ymin_position, self.ymax_position, 100)
            # xys = list(zip((xs-self.min_position)*scale, (ys-self.min_position)*scale))

            # self.track = rendering.make_polyline(xys)
            # self.track.set_linewidth(4)
            # self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            rat = rendering.make_circle(carheight)  # rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])#
            rat.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            rat.add_attr(self.cartrans)
            self.viewer.add_geom(rat)
            frontwheel = rendering.make_circle(carheight / 4)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            # backwheel = rendering.make_circle(carheight/2.5)
            # backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            # backwheel.add_attr(self.cartrans)
            # backwheel.set_color(.5, .5, .5)
            # self.viewer.add_geom(backwheel)
            flagx = (self.goal_position[0] - self.xmin_position) * scale_w
            flagy1 = (self.goal_position[1] - self.ymin_position) * scale_h
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state
        self.cartrans.set_translation((pos[0] - self.xmin_position) * scale_w, (pos[1] - self.ymin_position) * scale_h)
        self.cartrans.set_rotation(np.pi / 2 - self.action)
        # self.last_action = np.copy(self.action)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}  # control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _read_json_p(self):
        with open('sim_params.json', 'r') as fl:
            net_dict = json.load(fl)
        simtime = net_dict['simtime'] * 1000
        dt = net_dict['dt']
        self.steps_to_stop = simtime / dt
        self.max_tr_dur = net_dict['max_tr_dur'] / 1000.
        self.sim_env = net_dict['sim_env']
        self.env_limit_dic = net_dict['environment'][self.sim_env]
        if self.sim_env == 'openfield':
            self.xmin_position = float(self.env_limit_dic['xmin_position'])
            self.xmax_position = float(self.env_limit_dic['xmax_position'])
            self.ymin_position = float(self.env_limit_dic['ymin_position'])
            self.ymax_position = float(self.env_limit_dic['ymax_position'])
        else:
            print("environment {} undefined".format(self.sim_env))
        
        # calculate points from obstacle data for collision detection
        obs_dict = net_dict["environment"]["obstacles"]
        self.obstacle_list = []
        if obs_dict["flag"]:
            for center, vert, horiz in zip(obs_dict["centers"], obs_dict["vert_lengths"], obs_dict["horiz_lengths"]):
                delta_y = vert / 2. # Get the length and width 
                delta_x = horiz / 2.  # as distances from the center point
                
                ll = (center[0] - delta_x, center[1] - delta_y) # lower left
                lr = (center[0] + delta_x, center[1] - delta_y) # lower right
                ur = (center[0] + delta_x, center[1] + delta_y) # upper right
                ul = (center[0] - delta_x, center[1] + delta_y) # upper left
                
                # Note that the list of points needs to be given IN ORDER!
                # Otherwise, the diagonals of the rectangle will be treated as the obstacle borders
                self.obstacle_list.append([ll, lr, ur, ul])

        self.start_x = float(net_dict['start']['x_position'])
        self.start_y = float(net_dict['start']['y_position'])

        self.hide_goal = net_dict['goal']['hide_goal']
        self.reward_recep_field = net_dict['goal']['reward_recep_field']
        self.goal_x = net_dict['goal']['x_position']
        self.goal_y = net_dict['goal']['y_position']

        self.data_path = net_dict['data_path']
