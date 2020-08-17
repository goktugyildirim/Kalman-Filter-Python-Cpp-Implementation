import matplotlib.pyplot as plt
from kf import KF
import numpy as np


real_x = 0
real_v = 0.9

kalman = KF(initial_x = 10, initial_v = 2 ,accel_variance = 0.9)

mus = []
covs = []
meas_pos = []

dt = 0.1
NUM_STEPS = 1000

for step in range(NUM_STEPS):
    #det_before = np.linalg.det(kalman.cov())

    covs.append(kalman.cov())
    mus.append(kalman.mean())

    real_x = real_x + dt*real_v
    
    kalman.predict(dt)

    meas_variance = 0.5
    meas_value = real_x + np.random.randn() * np.sqrt(meas_variance) # real + error
    meas_pos.append(meas_value)
    kalman.update(meas_value, meas_variance)

    #det_after = np.linalg.det(kalman.cov())
    #print(det_before,det_after)



position = [element[0] for element in mus]
velocity = [element[1] for element in mus]


plt.plot(meas_pos, 'b') # measurements
plt.plot(position, 'r') #kalman filtreden geçirilmiş measurement values
plt.title("Position")
plt.show()
plt.figure()


plt.plot(velocity)
plt.title("Veloctiy")
plt.show()
plt.figure()