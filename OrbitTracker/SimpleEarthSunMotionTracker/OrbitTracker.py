import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from AstronomicalBody import Body

G = 6.67408e-8
N = 100000
Years = 1.99

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1, l1 at 2
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], VY[0], VZ[0])
sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

# The position and velocity of l1 and l2 depends on the position and velocity of earth. Thus, values will be generated here
earth_r = np.sqrt(X[0]**2 + Y[0]**2 + Z[0]**2)

l1_r = earth_r - np.cbrt(M[0]/(3*M[1])) * earth_r
l2_r = earth_r + np.cbrt(M[0]/(3*M[1])) * earth_r

l1_x = X[0] * l1_r/earth_r
l1_y = Y[0] * l1_r/earth_r
l1_z = Z[0] * l1_r/earth_r
l2_x = X[0] * l2_r/earth_r
l2_y = Y[0] * l2_r/earth_r
l2_z = Z[0] * l2_r/earth_r

l1_vx = VX[0]/earth_r * l1_r	
l1_vy = VY[0]/earth_r * l1_r
l1_vz = VZ[0]/earth_r * l1_r
l2_vx = VX[0]/earth_r * l2_r	
l2_vy = VY[0]/earth_r * l2_r
l2_vz = VZ[0]/earth_r * l2_r

print("l1_x: ", l1_x)
print("l1_y: ", l1_y)
print("l1_z: ", l1_z)
print("l1_vx: ", l1_vx)
print("l1_vy: ", l1_vy)
print("l1_vz: ", l1_vz)
print("")

l1 = Body(0, N, l1_x, l1_y, l1_z, l1_vx, l1_vy, l1_vz)
l2 = Body(0, N, l2_x, l2_y, l2_z, l2_vx, l2_vy, l2_vz)

bodies = np.array([earth, sun, l1, l2])


### Comment out these lines if not using Nasa Data ###
#M, X, Y, Z, VX, VY, VZ = np.loadtxt("Nasa_ICs.txt", unpack = True)
#M *= 10**3
#X *= 10**5; Y *= 10**5; Z *= 10**5; VX *= 10**5; VY *= 10**5; VZ *= 10**5;

#earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], VY[0], VZ[0])
#sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])
#l1 = Body(M[2], N, X[2], Y[2], Z[2], VX[2], VY[2], VZ[2])
#l2 = Body(M[3], N, X[3], Y[3], Z[3], VX[3], VY[3], VZ[3])
#jupiter = Body(M[4], N, X[4], Y[4], Z[4], VX[4], VY[4], VZ[4])
#moon = Body(M[5], N, X[5], Y[5], Z[5], VX[5], VY[5], VZ[5])

#bodies = np.array([earth, sun, l1, l2, jupiter, moon])
### End of Nasa Data ###


# calculate center of mass
x_center = 0
y_center = 0
z_center = 0
M_tot = 0
for body in bodies:
	x_center += body.mass * body.x[0]
	y_center += body.mass * body.y[0]
	z_center += body.mass * body.z[0]
	M_tot += body.mass

x_center /= M_tot
y_center /= M_tot
z_center /= M_tot

print("Center of mass initial: ", x_center, y_center, z_center)

# reset all positions according to center of mass
for body in bodies:
	body.x[0] -= x_center
	body.y[0] -= y_center
	body.z[0] -= z_center


### Define a function to track motion using leapfrog algorithm ###
def trackMotion(years, nsteps):
	print(earth.x[0]**3/G/M[1])
	#tf = years * 2*np.pi*np.sqrt(earth.x[0]**3/G/M[1])
	tf = years * 2*np.pi*np.sqrt(np.sqrt(earth.x[0]**2 + earth.y[0]**2 + earth.z[0]**2)**3/G/M[1])
	dt = tf / nsteps

	# Calculating initial acceleration
	for body in bodies:
		for otherBody in bodies:
			if body != otherBody:
				r = np.sqrt((body.x[0] - otherBody.x[0])**2 + (body.y[0] - otherBody.y[0])**2 + (body.z[0] - otherBody.z[0])**2)
				body.ax[0] += -G*otherBody.mass/r**2 * (body.x[0] - otherBody.x[0])/r
				body.ay[0] += -G*otherBody.mass/r**2 * (body.y[0] - otherBody.y[0])/r
				body.az[0] += -G*otherBody.mass/r**2 * (body.z[0] - otherBody.z[0])/r

	
	for i in range(1, nsteps):
		# Calculating position at half step
		for body in bodies:
			body.tempx = body.x[i-1] + body.vx[i-1]*0.5*dt
			body.tempy = body.y[i-1] + body.vy[i-1]*0.5*dt
			body.tempz = body.z[i-1] + body.vz[i-1]*0.5*dt

		# Calculatin acceleration at half step
		for body in bodies:
			for otherBody in bodies:
				if body != otherBody:
					r = np.sqrt((body.tempx - otherBody.tempx)**2 + (body.tempy - otherBody.tempy)**2 + (body.tempz - otherBody.tempz)**2)
					body.ax[i] += -G*otherBody.mass/r**2 * (body.tempx - otherBody.tempx)/r
					body.ay[i] += -G*otherBody.mass/r**2 * (body.tempy - otherBody.tempy)/r
					body.az[i] += -G*otherBody.mass/r**2 * (body.tempz - otherBody.tempz)/r

		# Calculating velocity
		for body in bodies:
			body.vx[i] = body.vx[i-1] + body.ax[i]*dt
			body.vy[i] = body.vy[i-1] + body.ay[i]*dt
			body.vz[i] = body.vz[i-1] + body.az[i]*dt

		# Calculating position at full step
		for body in bodies:
			body.x[i] = body.tempx + body.vx[i]*0.5*dt
			body.y[i] = body.tempy + body.vy[i]*0.5*dt
			body.z[i] = body.tempz + body.vz[i]*0.5*dt


trackMotion(Years, N)

plot1 = plt.figure(1)
plt.scatter(earth.x, earth.y, s=1, label='earth')
plt.scatter(sun.x, sun.y, s=1, label='sun')
plt.scatter(l1.x, l1.y, s=1, label='l1')
plt.scatter(l2.x, l2.y, s=1, label='l2')
plt.legend(loc=1)

plot2 = plt.figure(2)
graph1 = plt.axes(projection = '3d')
graph1.plot3D(earth.x, earth.y, earth.z)
graph1.plot3D(sun.x, sun.y, sun.z)
graph1.plot3D(l1.x, l1.y, l1.z)
graph1.plot3D(l2.x, l2.y, l2.z)

# calculate center of mass
x_center = 0
y_center = 0
z_center = 0
#M_tot = 0
for body in bodies:
	x_center += body.mass * body.x[-1]
	y_center += body.mass * body.y[-1]
	z_center += body.mass * body.z[-1]
	#M_tot += body.mass

x_center /= M_tot
y_center /= M_tot
z_center /= M_tot

print("Center of mass final: ", x_center, y_center, z_center)

#print("x initial: ", earth.x[0])
#print("x final: ", earth.x[-1])
#print("y initial: ", earth.y[0])
#print("y final: ", earth.y[-1])

plt.show()


