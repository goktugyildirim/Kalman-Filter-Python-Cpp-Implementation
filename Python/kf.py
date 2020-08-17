import numpy as np

class KF(object):
    def __init__(self, initial_x, initial_v, accel_variance):
        #Mean of state GRV
        self.x = np.array([initial_x, initial_v]).reshape((2,1))
        
        #Covariance of state GRV
        self.P = np.eye(2).reshape((2,2))
        
        self.accel_variance = accel_variance  


    def predict(self, dt):
        # x = F x
        # P = F P Ft + G Gt a
        F = np.array([[1,dt],[0,1]]).reshape((2,2))
        new_x = F.dot(self.x)

        G = np.array([0.5*dt**2,dt]).reshape((2,1))
        new_P = F.dot(self.P).dot(F.T) + G.dot(G.T) * self.accel_variance
        
        self.x = new_x
        self.P = new_P

    def update(self, meas_value, meas_variance):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([1,0]).reshape((1,2))
        z = np.array([meas_value])
        R = np.array([meas_variance])
        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R 
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self.x + K.dot(y)
        new_P = (np.eye(2)-K.dot(H)).dot(self.P)

        self.x = new_x
        self.P = new_P





    def mean(self):
        return self.x
    
    def cov(self):
        return self.P

    def pos(self):
        return self.x[0]


    def vel(self):
        return self.x[1]


