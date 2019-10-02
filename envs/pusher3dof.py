import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import xml.etree.ElementTree as et

import mujoco_py

class PusherEnv3DofEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self)
        self.reference_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           'assets/pusher_3dof.xml')
        mujoco_env.MujocoEnv.__init__(self, self.reference_path, frame_skip=5)

        self.model.stat.extent = 10

        # randomization
        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []
        self._locate_randomize_parameters()
        # self.checkMy = False

    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        end_effector = self.root.find(".//body[@name='distal_4']")
        self.wrist = end_effector.findall("./geom[@type='capsule']")
        self.tips = end_effector.findall(".//body[@name='tips_arm']/geom")
        self.object_body = self.root.find(".//body[@name='object']/geom")
        self.object_joints = self.root.findall(".//body[@name='object']/joint")

    def _update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def _create_xml(self):
        # TODO: I might speed this up, but I think is insignificant w.r.t to the model/sim creation...
        self._randomize_friction()
        self._randomize_damping()
        # self._randomize_size()

        return et.tostring(self.root, encoding='unicode', method='xml')

    # TODO: I'm making an assumption here that 3 places after the comma are good enough, are they?
    def _randomize_friction(self):
        frictionloss = self.dimensions[0].current_value

        for joint in self.object_joints:
            joint.set('frictionloss', '{:3f}'.format(frictionloss))

    def _randomize_damping(self):
        damping = self.dimensions[1].current_value
        for joint in self.object_joints:
            joint.set('damping', '{:3f}'.format(damping))

    def _randomize_size(self):
        size = self.dimensions[2].current_value
        # grabber
        grabber_width = size * 2
        self.wrist[0].set('fromto', '0 -{:3f} 0. 0.0 +{:3f} 0'.format(grabber_width, grabber_width))
        self.wrist[1].set('fromto', '0 -{:3f} 0. {:3f} -{:3f} 0'.format(grabber_width, grabber_width, grabber_width))
        self.wrist[2].set('fromto', '0 +{:3f} 0. {:3f} +{:3f} 0'.format(grabber_width, grabber_width, grabber_width))
        self.tips[0].set('pos', '{:3f} -{:3f} 0.'.format(grabber_width, grabber_width))
        self.tips[1].set('pos', '{:3f} {:3f} 0.'.format(grabber_width, grabber_width))

    def step(self, action):
        arm_dist = np.linalg.norm(self.get_body_com("object")[:2] - self.get_body_com("tips_arm")[:2])
        goal_dist = np.linalg.norm(self.get_body_com("object")[:2] - self.get_body_com("goal")[:2])

        # Reward from Soft Q Learning
        # action_cost = np.square(action).sum()
        reward = -goal_dist

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        done = False

        return ob, reward, done, {'arm_dist': arm_dist, 'goal_dist': goal_dist}

    def viewer_setup(self):
        # coords = [.7, -.5, 0]
        coords = [0.15, -0, -1000]
        for i in range(3):
            self.viewer.cam.lookat[i] = coords[i]
        # self.viewer.cam.trackbodyid = -1
        # self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.trackbodyid = -1      
        self.viewer.cam.distance = 4.25
        self.viewer.cam.lookat[2] = -0.2
        self.viewer.cam.elevation = -60
        print (self.viewer.cam.distance, self.viewer.cam.lookat,self.viewer.cam.elevation )
        # checkMy = True

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        # Original
        # object_ = np.random.uniform(low=[.3,-1.0], high=[1.2,-0.4])
        # goal = np.random.uniform(low=[.8,-1.2], high=[1.2,-0.8])
        
        while True:
            # NOW RUNNING: "HARDER*"
            object_ = np.random.uniform(low=[.4,-1.0], high=[1.2,-0.5])
            # object_ = np.random.uniform(low=[.5,-1.0], high=[1.2,-0.6])
            goal = np.random.uniform(low=[.8,-1.2], high=[1.2,-0.8])
            if np.linalg.norm(object_ - goal) > 0.45:
                break

        self.object = np.array(object_)
        self.goal = np.array(goal)

        qpos[-4:-2] = self.object
        qpos[-2:] = self.goal
        qvel = self.init_qvel
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # print (self.get_body_com("distal_4"))
        height, width = 64, 64
        camera_id = 0
        self._get_viewer('rgb_array').render(width, height)
        data = self._get_viewer('rgb_array').read_pixels(width, height, depth=False)
        return data
