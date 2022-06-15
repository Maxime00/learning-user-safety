import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
  def __init__(self, bullet_client, offset):
    self.bullet_client = bullet_client
    self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
    self.offset = np.array(offset)
    
    #print("offset=",offset)
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    self.legos=[]
    
    self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
    self.bullet_client.changeVisualShape(self.legos[0],-1,rgbaColor=[1,0,0,1])
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
    self.legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
    self.sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
    self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offset, flags=flags)
    self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)
    orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
    eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
    index = 0
    self.state = 0
    self.control_dt = 1./120.
    self.finger_target = 0
    self.gripper_height = 0.2
    #create a constraint to keep the fingers centered
    c = self.bullet_client.createConstraint(self.panda,
                       9,
                       self.panda,
                       10,
                       jointType=self.bullet_client.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
    self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
 
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      #print("info=",info)
      jointName = info[1]
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
        index=index+1
    self.prev_pos = np.array(self.bullet_client.getLinkState(self.panda, pandaEndEffectorIndex)[0])
    print(self.prev_pos)
    self.diffX = 0
    self.diffZ = 0
    self.diffY = 0
    self.diffG = 0
    self.finger_target = 0.02

    self.t = 0.
  def reset(self):
    pass

  def update_state(self):
    # self.diffX = 0
    # self.diffZ = 0
    # self.diffY = 0
    # self.diffG = 0
    keys = self.bullet_client.getKeyboardEvents()
    if len(keys)>0:
      for k,v in keys.items():
        if v&self.bullet_client.KEY_WAS_TRIGGERED:
          if (k== 65297):#'B3G_UP_ARROW'): #move up workspace
              print('up')
              self.diffX = 1
          if (k== 65295):#'B3G_LEFT_ARROW'):# move left workspace (Z)
              print('left')
              self.diffZ = -1
          if (k==65298):#'B3G_DOWN_ARROW'): #move down workspace
              print('down')
              self.diffX = -1
          if (k==65296):#'B3G_RIGHT_ARROW'): #move right workspace
              print('right')
              self.diffZ = 1
          if (k==97):#'B3G_DOWN_ARROW'): #move down workspace
              print('down')
              self.diffY = -1
          if (k==100):#'B3G_RIGHT_ARROW'): #move right workspace
              print('right')
              self.diffY = 1
          if (k==ord('c')):
              self.diffG = -1
          if (k==ord('x')):
              self.diffG = 1#+= 0.04

        if v&self.bullet_client.KEY_WAS_RELEASED:
          if (k== 65297):#'B3G_UP_ARROW'): #move up workspace
              print('up')
              self.diffX = 0
          if (k== 65295):#'B3G_LEFT_ARROW'):# move left workspace (Z)
              print('left')
              self.diffZ = 0
          if (k==65298):#'B3G_DOWN_ARROW'): #move down workspace
              print('down')
              self.diffX = 0
          if (k==65296):#'B3G_RIGHT_ARROW'): #move right workspace
              print('right')
              self.diffZ = 0
          if (k==97):#'B3G_DOWN_ARROW'): #move down workspace
              print('down')
              self.diffY = 0
          if (k==100):#'B3G_RIGHT_ARROW'): #move right workspace
              print('right')
              self.diffY = 0
          if (k==ord('c')):
              self.diffG = 0
          if (k==ord('x')):
              self.diffG = 0#+= 0.04
  def step(self):
    self.bullet_client.submitProfileTiming("step")
    self.update_state()
    t = self.t
    self.t += self.control_dt
    speed = 0.001
    pos = [self.prev_pos[0] + self.diffX*speed, self.prev_pos[1] + self.diffY *speed, self.prev_pos[2] + self.diffZ*speed]
    orn = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
    grip_speed = 0.05
    self.finger_target = self.finger_target + self.diffG * grip_speed 
    self.bullet_client.submitProfileTiming("IK")
    jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,pandaEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
    self.bullet_client.submitProfileTiming()
    for i in range(pandaNumDofs):
        self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, jointPoses[i],force=5 * 240.)
        #target for fingers
    for i in [9,10]:
      self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL,self.finger_target ,force= 10)
    self.prev_pos = pos

    self.bullet_client.submitProfileTiming()


