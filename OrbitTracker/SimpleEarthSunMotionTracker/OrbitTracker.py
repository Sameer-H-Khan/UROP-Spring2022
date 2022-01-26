import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from AstronomicalBody import Body

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1, l1 at 2
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

G = 6.67408e-8
N = 100000

earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], VY[0], VZ[0])
sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

# The position and velocity of l1 depends on the position and velocity of earth. Thus, values will be generated here
earth_r = np.sqrt(X[0]**2 + Y[0]**2 + Z[0]**2)

l1_r = earth_r - np.cbrt(M[0]/(3*M[1])) * earth_r

l1_x = X[0] * l1_r/earth_r
l1_y = Y[0] * l1_r/earth_r
l1_z = Z[0] * l1_r/earth_r

l1_vx = 0
l1_vy = 0
l1_vz = 0

if X[0] != 0:					# w = earth_v/earth_r
	l1_vx = VX[0]/X[0] * l1_x	# l1_v = w * l1_r
if Y[0] != 0:					# l1_v = (earth_v/earth_r) * l1_r
	l1_vy = VY[0]/Y[0] * l1_y
if Z[0] != 0:
	l1_vz = VZ[0]/Z[0] * l1_z

print("l1_x: ", l1_x)
print("l1_y: ", l1_y)
print("l1_z: ", l1_z)
print("l1_vx: ", l1_vx)
print("l1_vy: ", l1_vy)
print("l1_vz: ", l1_vz)

l1 = Body(0, N, l1_x, l1_y, l1_z, l1_vx, l1_vy, l1_vz)
l1 = Body(M[2], N, X[2], Y[2], Z[2], VX[2], VY[2], VZ[2])

bodies = np.array([earth, sun, l1])

def trackMotion(years, nsteps):
	tf = years * 2*np.pi*np.sqrt(earth.x[0]**3/G/M[1])
	dt = tf / nsteps

	for body in bodies:
		for otherBody in bodies:
			if body != otherBody:
				r = np.sqrt((body.x[0] - otherBody.x[0])**2 + (body.y[0] - otherBody.y[0])**2 + (body.z[0] - otherBody.z[0])**2)
				body.ax[0] += -G*otherBody.mass/r**2 * (body.x[0] - otherBody.x[0])/r
				body.ay[0] += -G*otherBody.mass/r**2 * (body.y[0] - otherBody.y[0])/r
				body.az[0] += -G*otherBody.mass/r**2 * (body.z[0] - otherBody.z[0])/r

	for i in range(1, nsteps):
		for body in bodies:
			body.tempx = body.x[i-1] + body.vx[i-1]*0.5*dt
			body.tempy = body.y[i-1] + body.vy[i-1]*0.5*dt
			body.tempz = body.z[i-1] + body.vz[i-1]*0.5*dt

		for body in bodies:
			for otherBody in bodies:
				if body != otherBody:
					r = np.sqrt((body.tempx - otherBody.tempx)**2 + (body.tempy - otherBody.tempy)**2 + (body.tempz - otherBody.tempz)**2)
					body.ax[i] += -G*otherBody.mass/r**2 * (body.tempx - otherBody.tempx)/r
					body.ay[i] += -G*otherBody.mass/r**2 * (body.tempy - otherBody.tempy)/r
					body.az[i] += -G*otherBody.mass/r**2 * (body.tempz - otherBody.tempz)/r

		for body in bodies:
			body.vx[i] = body.vx[i-1] + body.ax[i]*dt
			body.vy[i] = body.vy[i-1] + body.ay[i]*dt
			body.vz[i] = body.vz[i-1] + body.az[i]*dt

		for body in bodies:
			body.x[i] = body.tempx + body.vx[i]*0.5*dt
			body.y[i] = body.tempy + body.vy[i]*0.5*dt
			body.z[i] = body.tempz + body.vz[i]*0.5*dt


trackMotion(1, N)

plot1 = plt.figure(1)
plt.scatter(earth.x, earth.y, s=1)
plt.scatter(sun.x, sun.y, s=1)
plt.scatter(l1.x, l1.y, s=1)

plot2 = plt.figure(2)
graph1 = plt.axes(projection = '3d')
graph1.plot3D(earth.x, earth.y, earth.z)
graph1.plot3D(sun.x, sun.y, sun.z)
graph1.plot3D(l1.x, l1.y, l1.z)

print("x initial: ", earth.x[0])
print("x final: ", earth.x[-1])
print("y initial: ", earth.y[0])
print("y final: ", earth.y[-1])

plt.show()


