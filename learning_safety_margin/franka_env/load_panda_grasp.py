import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import panda_sim_grasp_controller as panda_sim

#video requires ffmpeg available in path
createVideo=False

if createVideo:
	p.connect(p.GUI, options="--mp4=\"pybullet_grasp.mp4\", --mp4fps=240")
else:
	p.connect(p.GUI)

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35,-0.13,0])
p.setAdditionalSearchPath(pd.getDataPath())

timeStep=1./120.#240.
p.setTimeStep(timeStep)
p.setGravity(0,-9.8,0)

panda = panda_sim.PandaSim(p,[0,0,0])
logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
panda.bullet_client.submitProfileTiming("start")
for i in range (100000):
	panda.bullet_client.submitProfileTiming("full_step")
	panda.step()
	p.stepSimulation()
	if createVideo:
		p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
	time.sleep(timeStep)
	panda.bullet_client.submitProfileTiming()
panda.bullet_client.submitProfileTiming()
panda.bullet_client.stopStateLogging(logId)
