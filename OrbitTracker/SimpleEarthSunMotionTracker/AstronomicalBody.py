import numpy as np

class Body:
    
    tempx = 0
    tempy = 0
    tempz = 0

    def __init__(self, mass, numberOfSteps, xi, yi, zi, vxi, vyi, vzi):
        self.mass = mass

        self.x = np.zeros(numberOfSteps)
        self.y = np.zeros(numberOfSteps)
        self.z = np.zeros(numberOfSteps)

        self.vx = np.zeros(numberOfSteps)
        self.vy = np.zeros(numberOfSteps)
        self.vz = np.zeros(numberOfSteps)

        self.ax = np.zeros(numberOfSteps)
        self.ay = np.zeros(numberOfSteps)
        self.az = np.zeros(numberOfSteps)

        self.x[0] = xi; self.y[0] = yi; self.z[0] = zi;
        self.vx[0] = vxi; self.vy[0] = vyi; self.vz[0] = vzi;
