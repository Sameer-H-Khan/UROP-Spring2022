import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from AstronomicalBody import Body

G = 6.67408e-8
N = 5000
Years = 1

earth_equ_radius = 637816000

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1, l1 at 2
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

vy_init = np.sqrt(G*M[1]/X[0])
print(vy_init/1e5)
VY[0] = vy_init
#M[0] = 0.1*M[1]
earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

# The position and velocity of l1 and l2 depends on the position and velocity of earth. Thus, values will be generated here
earth_r = np.sqrt((X[0]-X[1])**2 + (Y[0]-Y[1])**2 + (Z[0]-Z[1])**2)

#l1_r = earth_r - np.cbrt(M[0]/(3*M[1])) * earth_r
#l2_r = earth_r + np.cbrt(M[0]/(3*M[1])) * earth_r


# Acceleration is a function of x. This is the function to minimize
#def acc_on_point(p_x):
#	acc_earth_on_point = -G*earth.x[0]/(p_x - earth.x[0])**2
#	acc_sun_on_point =  -G*sun.x[0]/(p_x - sun.x[0])**2
#	acc_tot = np.abs(acc_earth_on_point + acc_sun_on_point)
#	return acc_tot

#r_start = earth_r - np.cbrt(M[0]/(3*M[1])) * earth_r	# Starting guess for the value of x that'll give a minimum acceleration
#result = opt.minimize(acc_on_point, r_start)
#l1x_calc = result.x			# x value that gives minimum acceleration
#l1acc_calc = result.fun		# minimum acceleration-- should be ~0

#print("")
#print("=================================")
#print("l1x:\t", l1x_calc)
#print("l1acc:\t", l1acc_calc)

#l1_x = l1x_calc
#l1_y = Y[0] * l1x_calc/earth_r
#l1_z = Z[0] * l1x_calc/earth_r
#l2_x = X[0] * l2_r/earth_r
#l2_y = Y[0] * l2_r/earth_r
#l2_z = Z[0] * l2_r/earth_r

#l1_vx = VX[0]/earth_r * l1x_calc	
#l1_vy = VY[0]/earth_r * l1x_calc
#l1_vz = VZ[0]/earth_r * l1x_calc
#l2_vx = VX[0]/earth_r * l2_r	
#l2_vy = VY[0]/earth_r * l2_r
#l2_vz = VZ[0]/earth_r * l2_r

#print("l1_x: ", l1_x)
#print("l1_y: ", l1_y)
#print("l1_z: ", l1_z)
#print("l1_vx: ", l1_vx)
#print("l1_vy: ", l1_vy)
#print("l1_vz: ", l1_vz)
#print("")

# Manual l1 calculations for general cases where not aligned on x axis in this hidden block
#l1_x = X[0] * l1_r/earth_r
#l1_y = Y[0] * l1_r/earth_r
#l1_z = Z[0] * l1_r/earth_r
#l2_x = X[0] * l2_r/earth_r
#l2_y = Y[0] * l2_r/earth_r
#l2_z = Z[0] * l2_r/earth_r

#l1_vx = VX[0]/earth_r * l1_r	
#l1_vy = VY[0]/earth_r * l1_r
#l1_vz = VZ[0]/earth_r * l1_r
#l2_vx = VX[0]/earth_r * l2_r	
#l2_vy = VY[0]/earth_r * l2_r
#l2_vz = VZ[0]/earth_r * l2_r

#print("l1_x: ", l1_x)
#print("l1_y: ", l1_y)
#print("l1_z: ", l1_z)
#print("l1_vx: ", l1_vx)
#print("l1_vy: ", l1_vy)
#print("l1_vz: ", l1_vz)
#print("")

#l1 = Body(0, N, l1_x, l1_y, l1_z, l1_vx, l1_vy, l1_vz)
#l2 = Body(0, N, l2_x, l2_y, l2_z, l2_vx, l2_vy, l2_vz)

bodies = np.array([earth, sun])

### Comment out these lines if not using Nasa Data ### Data taken from 2022-01-12
#M, X, Y, Z, VX, VY, VZ = np.loadtxt("Nasa_ICs.txt", unpack = True)
#M *= 10**3
#X *= 10**5; Y *= 10**5; Z *= 10**5; VX *= 10**5; VY *= 10**5; VZ *= 10**5;
#print(np.sqrt(VX[0]**2 + VY[0]**2 + VZ[0]**2))

#earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], VY[0], VZ[0])
#sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])
#l1 = Body(M[2], N, X[2], Y[2], Z[2], VX[2], VY[2], VZ[2])
#l2 = Body(M[3], N, X[3], Y[3], Z[3], VX[3], VY[3], VZ[3])
#jupiter = Body(M[4], N, X[4], Y[4], Z[4], VX[4], VY[4], VZ[4])
#moon = Body(M[5], N, X[5], Y[5], Z[5], VX[5], VY[5], VZ[5])

#bodies = np.array([earth, sun, l1, l2, jupiter, moon])
### End of Nasa Data ###

# calculate center of mass
#x_center = 0
#y_center = 0
#z_center = 0
#M_tot = 0
#for body in bodies:
#	x_center += body.mass * body.x[0]
#	y_center += body.mass * body.y[0]
#	z_center += body.mass * body.z[0]
#	M_tot += body.mass

#x_center /= M_tot
#y_center /= M_tot
#z_center /= M_tot

#print("Center of mass initial: ", x_center, y_center, z_center)

def centerMassAndVelocity(i):
	x_center = 0
	y_center = 0
	z_center = 0

	vx_center = 0
	vy_center = 0
	vz_center = 0

	M_tot = 0
	for body in bodies:
		x_center += body.mass * body.x[i]
		y_center += body.mass * body.y[i]
		z_center += body.mass * body.z[i]

		vx_center += body.mass * body.vx[i]
		vy_center += body.mass * body.vy[i]
		vz_center += body.mass * body.vz[i]
		M_tot += body.mass

	x_center /= M_tot
	y_center /= M_tot
	z_center /= M_tot

	vx_center /= M_tot
	vy_center /= M_tot
	vz_center /= M_tot

	return x_center, y_center, z_center, vx_center, vy_center, vz_center

xc, yc, zc, vxc, vyc, vzc = centerMassAndVelocity(0)

# reset all positions according to center of mass
for body in bodies:
	body.x[0] -= xc
	body.y[0] -= yc
	body.z[0] -= zc

	body.vx[0] -= vxc
	body.vy[0] -= vyc
	body.vz[0] -= vzc

# Center velocity
#vx_center = 0
#vy_center = 0
#vz_center = 0
##M_tot = 0
#for body in bodies:
#	vx_center += body.mass * body.vx[0]
#	vy_center += body.mass * body.vy[0]
#	vz_center += body.mass * body.vz[0]
#	#M_tot += body.mass

#vx_center /= M_tot
#vy_center /= M_tot
#vz_center /= M_tot

#print("Center of mass initial: ", x_center, y_center, z_center)

# reset velocities according to center
#for body in bodies:
#	body.vx[0] -= vx_center
#	body.vy[0] -= vy_center
#	body.vz[0] -= vz_center

xc, yc, zc, vxc, vyc, vzc = centerMassAndVelocity(0)

print("Center of mass initial: ", xc, yc, zc)
print("Center of velocity initial: ", vxc, vyc, vzc)



def force_on_l1(h):
	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)
	lhs = -G*sun.mass/(r_es - h)**2 + G*earth.mass/h**2
	rhs = -G*sun.mass/r_es**3 * (r_es - h)
	return lhs - rhs

def force_on_l2(h):
	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)
	lhs = G*sun.mass/(r_es + h)**2 + G*earth.mass/h**2
	rhs = G*sun.mass/r_es**3 * (r_es + h)
	return lhs - rhs


### Define a function to track motion using leapfrog algorithm ###
def trackMotion(years, nsteps):
	#tf = years * 2*np.pi*np.sqrt(earth.x[0]**3/G/M[1])
	tf = years * 2*np.pi*np.sqrt(np.sqrt((earth.x[0] - sun.x[0])**2 + (earth.y[0] - sun.y[0])**2 + (earth.z[0] - sun.z[0])**2)**3/G/M[1])
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

		# Calculating acceleration at half step
		for body in bodies:
			for otherBody in bodies:
				if body != otherBody and otherBody.mass != 0:
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


def distance_between_l1_start_and_end(l1Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	#l1_vy = l1_x_initialGuess * np.sqrt(G*sun.mass/r_es**3)
	l1_vy = l1Start * np.sqrt(G*sun.mass/r_es**3)
	l1 = Body(0, N, l1Start, 0, 0, 0, l1_vy, 0)

	bodies = np.array([earth, sun, l1])

	trackMotion(1, N)
	r_x = l1.x[-1] - earth.x[-1]
	r_y = l1.y[-1] - earth.y[-1]
	r_z = l1.z[-1] - earth.z[-1]
	return np.sqrt(r_x**2 + r_y**2 + r_z**2)


def distance_between_l2_start_and_end(l2Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	#l2_vy = l2_x_initialGuess * np.sqrt(G*sun.mass/r_es**3)
	l2_vy = l2Start * np.sqrt(G*sun.mass/r_es**3)
	l2 = Body(0, N, l2Start, 0, 0, 0, l2_vy, 0)

	bodies = np.array([earth, sun, l2])

	trackMotion(1, N)
	r_x = l2.x[-1] - earth.x[-1]
	r_y = l2.y[-1] - earth.y[-1]
	r_z = l2.z[-1] - earth.z[-1]
	return np.sqrt(r_x**2 + r_y**2 + r_z**2)


## A more specific version of trackMotion which also tracks the distance between earth and lagrange points throughout the orbit
## Ideally, lagrangeGuess should be replaced with a vector with x, y, and z components, and find the difference in vectors when 
## filling earthToLagrangeDrifts 
def trackMotionAndDrifts(years, nsteps, bodies, lagrangeBody, lagrangeGuess):
	#tf = years * 2*np.pi*np.sqrt(earth.x[0]**3/G/M[1])
	tf = years * 2*np.pi*np.sqrt(np.sqrt((earth.x[0] - sun.x[0])**2 + (earth.y[0] - sun.y[0])**2 + (earth.z[0] - sun.z[0])**2)**3/G/M[1])
	dt = tf / nsteps

	earthToLagrangeDrifts = np.zeros(nsteps)

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

		# Calculating acceleration at half step
		for body in bodies:
			for otherBody in bodies:
				if body != otherBody and otherBody.mass != 0:
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

		# Find the difference in position between earth and l1 and earth and l2
		earthToLagrangeDrifts[i] = np.sqrt((lagrangeBody.x[i] - earth.x[i])**2 + (lagrangeBody.y[i] - earth.y[i])**2 + (lagrangeBody.z[i] - earth.z[i])**2) - lagrangeGuess

	return earthToLagrangeDrifts



## A more specific version of trackMotion which also tracks whether or not the object at the lagrange point
## blocks a ray of light from the center of the sun to the earth. Returns an array of bits. 0 if the lagrange
## point did not block the ray of light. 1 if the lagrange point did block the ray of light
########## NOT SURE IF THIS WILL WORK IN 3D ##########
def trackMotionAndRays(years, nsteps, bodies, earthBody, lagrangeBody, lagrangeGuess):
	#tf = years * 2*np.pi*np.sqrt(earth.x[0]**3/G/M[1])
	tf = years * 2*np.pi*np.sqrt(np.sqrt((earth.x[0] - sun.x[0])**2 + (earth.y[0] - sun.y[0])**2 + (earth.z[0] - sun.z[0])**2)**3/G/M[1])
	dt = tf / nsteps

	#raysBlockedByLagrange = np.zeros(nsteps)
	#raysBlockedByLagrange[0] = 1
	angles = np.zeros(nsteps)

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

		# Calculating acceleration at half step
		for body in bodies:
			for otherBody in bodies:
				if body != otherBody and otherBody.mass != 0:
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
		for body in bodies: # func(bodies, timestep, currtime)
			body.x[i] = body.tempx + body.vx[i]*0.5*dt
			body.y[i] = body.tempy + body.vy[i]*0.5*dt
			body.z[i] = body.tempz + body.vz[i]*0.5*dt

		# Find whether or not the object at the lagrange point blocked a ray from the sun to the earth
		#distance_from_lagrange_to_sun = np.sqrt((lagrangeBody.x[i] - sun.x[i])**2 + (lagrangeBody.y[i] - sun.y[i])**2 + (lagrangeBody.z[i] - sun.z[i])**2)
		distance_from_earth_to_sun = np.sqrt((earthBody.x[i] - sun.x[i])**2 + (earthBody.y[i] - sun.y[i])**2 + (earthBody.z[i] - sun.z[i])**2)
		distance_from_earth_to_l1 = np.sqrt((earthBody.x[i] - lagrangeBody.x[i])**2 + (earthBody.y[i] - lagrangeBody.y[i])**2 + (earthBody.z[i] - lagrangeBody.z[i])**2)

		r_l_hatX = (lagrangeBody.x[i] - earthBody.x[i]) / distance_from_earth_to_l1
		r_l_hatY = (lagrangeBody.y[i] - earthBody.y[i]) / distance_from_earth_to_l1
		r_l_hatZ = (lagrangeBody.z[i] - earthBody.z[i]) / distance_from_earth_to_l1

		r_s_hatX = (sun.x[i] - earthBody.x[i]) / distance_from_earth_to_sun
		r_s_hatY = (sun.y[i] - earthBody.y[i]) / distance_from_earth_to_sun
		r_s_hatZ = (sun.z[i] - earthBody.z[i]) / distance_from_earth_to_sun

		rl_dot_rs = r_l_hatX * r_s_hatX + r_l_hatY * r_s_hatY +  r_l_hatZ * r_s_hatZ
		#if np.abs(rl_dot_re) > 1:
			#print("dot product exceeds 1: ", rl_dot_re)

		theta = np.arccos(max(min(rl_dot_rs, 1), -1))
		#print("theta: ", theta)
		#theta_crit = np.arcsin(earth_equ_radius/distance_from_earth_to_sun)

		#if theta < theta_crit:
		#	raysBlockedByLagrange[i] = 1
		#else:
		#	raysBlockedByLagrange[i] = 0

		angles[i] = theta

	#return raysBlockedByLagrange
	return angles



def mean_l1_drifts_from_earth(l1Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	l1_vy = l1Start * np.sqrt(G*sun.mass/r_es**3)
	l1 = Body(0, N, l1Start, 0, 0, 0, l1_vy, 0)

	bodies = np.array([earth, sun, l1])
	earthToL1Drifts = trackMotionAndDrifts(Years, N, bodies, l1, l1Start)
	return np.mean(earthToL1Drifts)


def mean_l2_drifts_from_earth(l2Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	l2_vy = l2Start * np.sqrt(G*sun.mass/r_es**3)
	l2 = Body(0, N, l2Start, 0, 0, 0, l2_vy, 0)

	bodies = np.array([earth, sun, l2])
	earthToL2Drifts = trackMotionAndDrifts(Years, N, bodies, l2, l2Start)
	return np.mean(earthToL2Drifts)


def sum_l1_blocks_of_light(l1Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	l1_vy = l1Start * np.sqrt(G*sun.mass/r_es**3)
	l1 = Body(0, N, l1Start, 0, 0, 0, l1_vy, 0)

	bodies = np.array([earth, sun, l1])
	raysBlockedByLagrange = trackMotionAndRays(1, 5000, bodies, earth, l1, l1Start)							# Altering opimization paramters
	#print(-1 * np.mean(raysBlockedByLagrange))
	#return -1 * np.sum(raysBlockedByLagrange)
	return np.mean(raysBlockedByLagrange)


hill_l1 = opt.brentq(force_on_l1, 1e+9, 1e+12)
print("")
print("hill_l1:\t", hill_l1)
print("sol/hill_l1.x:\t", hill_l1/earth.x[0])
print("earth.x - hill_l1:\t", earth.x[0] - hill_l1)

#hill_l2 = opt.brentq(force_on_l2, 1e+9, 1e+12)
#print("")
#print("hill_l2:\t", hill_l2)
#print("sol/hill_l2.x:\t", hill_l2/earth.x[0])
#print("earth.x + hill_l2:\t", earth.x[0] - hill_l2)

l1_x_initialGuess = earth.x[0] - hill_l1
#l2_x_initialGuess = earth.x[0] + hill_l2

#l1_x_final = opt.brentq(distance_between_l1_start_and_end, l1_x_initialGuess - 0.2*l1_x_initialGuess, l1_x_initialGuess + 0.2*l1_x_initialGuess)
#l1_x_final = opt.brentq(mean_l1_drifts_from_earth, l1_x_initialGuess - 0.1*l1_x_initialGuess, l1_x_initialGuess + 0.1*l1_x_initialGuess) # works pretty well

#result = opt.minimize(sum_l1_blocks_of_light, l1_x_initialGuess)
#l1_x_final = result.x
#number_of_steps_blocked = -1 * result.fun
#number_of_steps_unblocked = N - number_of_steps_blocked
#print("\nRatio of steps blocked to steps unblocked (higher is better):\t", number_of_steps_blocked/number_of_steps_unblocked, "\n")

print("\n******** Testing sum_l1_blocks_of_light ***********")
print(sum_l1_blocks_of_light(0.995*l1_x_initialGuess))
print(sum_l1_blocks_of_light(l1_x_initialGuess))
print(sum_l1_blocks_of_light(1.007*l1_x_initialGuess), "\n")

res = opt.brent(sum_l1_blocks_of_light, brack=(0.995*l1_x_initialGuess, l1_x_initialGuess, 1.007*l1_x_initialGuess), tol=1e-12, full_output=True)
#print(res)
#quit()
l1_x_final = res[0]
avg_angle = res[1]
print("l1 x final: ", l1_x_final)
print("best average angle: ", avg_angle)
#quit()
#l2_x_final = opt.brentq(distance_between_l2_start_and_end, l2_x_initialGuess - 0.2*l2_x_initialGuess, l2_x_initialGuess + 0.2*l2_x_initialGuess)
#l2_x_final = opt.brentq(mean_l2_drifts_from_earth, l2_x_initialGuess - 0.1*l2_x_initialGuess, l2_x_initialGuess + 0.1*l2_x_initialGuess)



# w_earth = sqrt(G * earth.mass/r_earth_sun**3)
# w_l1 = w_l2 = w_earth
# v = rw
r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)
l1_vy = l1_x_final * np.sqrt(G*sun.mass/r_es**3)
#l2_vy = l2_x_final * np.sqrt(G*sun.mass/r_es**3)

#l1_vy = l1_x * np.sqrt(G*sun.mass/(l1_x - sun.x[0])**3) # Should give the same values, but it's not
#l2_vy = l2_x * np.sqrt(G*sun.mass/(l2_x - sun.x[0])**3)

earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])
l1 = Body(0, N, l1_x_final, 0, 0, 0, l1_vy, 0)
#l2 = Body(0, N, l2_x_final, 0, 0, 0, l2_vy, 0)

bodies = np.array([earth, sun, l1])

# Adding random tracer particles around l1
for i in range(100):
	radius = 100000
	r = radius * ( np.random.random()**(1./3.) )  
	phi = np.random.uniform(0,2*np.pi) 
	costheta = np.random.uniform(-1,1) 
	theta = np.arccos(costheta) 
	x = r * np.sin(theta) * np.cos(phi) + l1_x_final
	y = r * np.sin(theta) * np.sin(phi)
	z = r * np.cos(theta)

	point = Body(0, N, x, y, z, 0, l1_vy, 0)
	bodies = np.append(bodies, point)

trackMotion(Years, N)

blocks = np.zeros(N)
for b in bodies:
	for i in range(N): 
		if b != earth and b != sun and b != l1 and not b.covered:
			# Find whether or not the tracer particle blocked a ray from the sun to the earth
			distance_from_earth_to_sun = np.sqrt((earth.x[i] - sun.x[i])**2 + (earth.y[i] - sun.y[i])**2 + (earth.z[i] - sun.z[i])**2)
			distance_from_earth_to_tracer = np.sqrt((earth.x[i] - b.x[i])**2 + (earth.y[i] - b.y[i])**2 + (earth.z[i] - b.z[i])**2)

			r_l_hatX = (b.x[i] - earth.x[i]) / distance_from_earth_to_tracer
			r_l_hatY = (b.y[i] - earth.y[i]) / distance_from_earth_to_tracer
			r_l_hatZ = (b.z[i] - earth.z[i]) / distance_from_earth_to_tracer

			r_s_hatX = (sun.x[i] - earth.x[i]) / distance_from_earth_to_sun
			r_s_hatY = (sun.y[i] - earth.y[i]) / distance_from_earth_to_sun
			r_s_hatZ = (sun.z[i] - earth.z[i]) / distance_from_earth_to_sun

			rl_dot_rs = r_l_hatX * r_s_hatX + r_l_hatY * r_s_hatY +  r_l_hatZ * r_s_hatZ
			#if np.abs(rl_dot_re) > 1:
				#print("dot product exceeds 1: ", rl_dot_re)

			theta = np.arccos(max(min(rl_dot_rs, 1), -1))

			if theta < 6.97e10 / distance_from_earth_to_sun:
				blocks[i] += 1
				#total_SA += SA_of_tracer

			#for b2 in bodies:
			#	if b2 != earth and b2 != sun and b2 != l1 and b2 != b:
			#		# if b is overlapping other tracer, for now, don't add its SA to the total
			#		distance_from_earth_to_tracer2 = np.sqrt((earth.x[i] - b2.x[i])**2 + (earth.y[i] - b2.y[i])**2 + (earth.z[i] - b2.z[i])**2)

			#		r_b2_hatX = (b2.x[i] - earth.x[i]) / distance_from_earth_to_tracer2
			#		r_b2_hatY = (b2.y[i] - earth.y[i]) / distance_from_earth_to_tracer2
			#		r_b2_hatZ = (b2.z[i] - earth.z[i]) / distance_from_earth_to_tracer2

			#		rl_dot_rl2 = r_l_hatX * r_b2_hatX + r_l_hatY * r_b2_hatY +  r_l_hatZ * r_b2_hatZ

			#		theta2 = np.arccos(max(min(rl_dot_rs, 1), -1))

			#		if theta2 < SA_of_tracer / distance_from_earth_to_tracer2: # don't know if this is right or not
			#			b2.covered = TRUE

total_SA = 0;
SA_of_tracer = np.pi * 10**(-8)
SA_blocked_at_step = np.zeros(N)
j = 0
for b in bodies:
	for i in range(1, N, 100):
		if b != earth and b != sun and b != l1 and not b.covered:
			for b2 in bodies:
				if b2 != earth and b2 != sun and b2 != l1 and b2 != b:
						# Find whether or not the tracer particle blocked a ray from the sun to the earth
						distance_from_earth_to_sun = np.sqrt((earth.x[i] - sun.x[i])**2 + (earth.y[i] - sun.y[i])**2 + (earth.z[i] - sun.z[i])**2)
						distance_from_earth_to_tracer = np.sqrt((earth.x[i] - b.x[i])**2 + (earth.y[i] - b.y[i])**2 + (earth.z[i] - b.z[i])**2)

						r_l_hatX = (b.x[i] - earth.x[i]) / distance_from_earth_to_tracer
						r_l_hatY = (b.y[i] - earth.y[i]) / distance_from_earth_to_tracer
						r_l_hatZ = (b.z[i] - earth.z[i]) / distance_from_earth_to_tracer

						r_s_hatX = (sun.x[i] - earth.x[i]) / distance_from_earth_to_sun
						r_s_hatY = (sun.y[i] - earth.y[i]) / distance_from_earth_to_sun
						r_s_hatZ = (sun.z[i] - earth.z[i]) / distance_from_earth_to_sun

						rl_dot_rs = r_l_hatX * r_s_hatX + r_l_hatY * r_s_hatY +  r_l_hatZ * r_s_hatZ

						theta = np.arccos(max(min(rl_dot_rs, 1), -1))

						if theta < 6.97e10 / distance_from_earth_to_sun:
							SA_blocked_at_step[i] += SA_of_tracer

						# if b is overlapping other tracers, mark them as covered
						distance_from_earth_to_tracer2 = np.sqrt((earth.x[i] - b2.x[i])**2 + (earth.y[i] - b2.y[i])**2 + (earth.z[i] - b2.z[i])**2)

						r_b2_hatX = (b2.x[i] - earth.x[i]) / distance_from_earth_to_tracer2
						r_b2_hatY = (b2.y[i] - earth.y[i]) / distance_from_earth_to_tracer2
						r_b2_hatZ = (b2.z[i] - earth.z[i]) / distance_from_earth_to_tracer2

						rl_dot_rb2 = r_l_hatX * r_b2_hatX + r_l_hatY * r_b2_hatY +  r_l_hatZ * r_b2_hatZ

						theta2 = np.arccos(max(min(rl_dot_rs, 1), -1))
						j += 1
						if theta2 < SA_of_tracer / distance_from_earth_to_tracer2: # don't know if this is right or not
							b2.covered = True
						


print("Printing blocks:")
print(blocks[::100])

print("")
print("Projected surface area of the earth: ", np.pi*630000000**2)
print("Estimated surface area blocked by tracers: ", SA_blocked_at_step)



xc, yc, zc, vxc, vyc, vzc = centerMassAndVelocity(-1)
print("Center of mass final: ", xc, yc, zc)
print("Center of velocity final: ", vxc, vyc, vzc)

print(" ")
print("###############")
print("Earth initial position:\t", earth.x[0], earth.y[0], earth.z[0])
print("L1 initial position:\t", l1.x[0], l1.y[0], l1.z[0])
#print("L2 initial position:\t", l2.x[0], l2.y[0], l2.z[0])

print("###############")
print("Earth final position:\t", earth.x[-1], earth.y[-1], earth.z[-1])
print("L1 final position:\t", l1.x[-1], l1.y[-1], l1.z[-1])
#print("L2 final position:\t", l2.x[-1], l2.y[-1], l2.z[-1])

print("\nTracer particles' starting positions")


#print(" ")
#print("**********Initially*********")
#print("L1 Acceleration:\t", l1.ax[0], l1.ay[0], l1.az[0])
#print("L2 Acceleration:\t", l2.ax[0], l2.ay[0], l2.az[0])
#print("Earth Acceleration:\t", earth.ax[0], earth.ay[0], earth.az[0])
#print(" ")
#print("L1 angular velocity:\t", l1.vy[0]/l1.x[0])
#print("L2 angular velocity:\t", l2.vy[0]/l2.x[0])
#print("Earth angular velocity:\t", earth.vy[0]/earth.x[0])
#print(" ")
#print("L1 centripetal acc:\t", l1.vy[0]**2/l1.x[0])
#print("L1 centripetal acc:\t", l2.vy[0]**2/l2.x[0])
#print("Earth centripetal acc:\t", earth.vy[0]**2/earth.x[0])


#print(" ")
#print("**********After ", Years, " years *********")
#print("L1 Acceleration:\t", l1.ax[-1], l1.ay[-1], l1.az[-1])
#print("L2 Acceleration:\t", l2.ax[-1], l2.ay[-1], l2.az[-1])
#print("Earth Acceleration:\t", earth.ax[-1], earth.ay[-1], earth.az[-1])
#print(" ")
#print("L1 angular velocity:\t", l1.vy[-1]/l1.x[-1])
#print("L2 angular velocity:\t", l2.vy[-1]/l2.x[-1])
#print("Earth angular velocity:\t", earth.vy[-1]/earth.x[-1])
#print(" ")
#print("L1 centripetal acc:\t", l1.vy[-1]**2/l1.x[-1])
#print("L1 centripetal acc:\t", l2.vy[-1]**2/l2.x[-1])
#print("Earth centripetal acc:\t", earth.vy[-1]**2/earth.x[-1])

#earth_radius_arr = np.sqrt(earth.x**2 + earth.y**2 + earth.z**2)
#earth_r_max = np.amax(earth_radius_arr)
#earth_r_min = np.amin(earth_radius_arr)
#print("")
#print("Earth radius max:\t", earth_r_max)
#print("Earth radius min:\t", earth_r_min)
#print("Max - min:\t", earth_r_max - earth_r_min)
#print("Hill radius:\t", np.cbrt(M[0]/(3*M[1])) * earth_r)
#print("Max - min is ", (earth_r_max - earth_r_min)/(np.cbrt(M[0]/(3*M[1])) * earth_r), " of hill radius")


#print("")

#earth_v_dot_rhat = earth.vx * earth.x + earth.vy*earth.y + earth.vz + earth.z / earth.getR()
#temp_vx = earth.vx - earth_v_dot_rhat * earth.x / earth.getR()
#temp_vy = earth.vy - earth_v_dot_rhat * earth.y / earth.getR()
#temp_vz = earth.vz - earth_v_dot_rhat * earth.z / earth.getR()
#earth_v_tang = np.sqrt(temp_vx**2 + temp_vy**2 + temp_vz**2)

#earth_angular = earth_v_tang / earth.getR()
#print(earth_angular * 2*np.pi*np.sqrt(np.sqrt(earth.x[0]**2 + earth.y[0]**2 + earth.z[0]**2)**3/G/M[1]))
##earth_angular = np.sqrt(earth.vx**2 + earth.vy**2 + earth.vz**2)/(np.sqrt(earth.x**2 + earth.y**2 + earth.z**2))
#print("Earth_angular min: ", np.min(earth_angular))
#print("Earth_angular max: ", np.max(earth_angular))
#print("")
#l1_angular = np.sqrt(l1.vx**2 + l1.vy**2 + l1.vz**2)/(np.sqrt(l1.x**2 + l1.y**2 + l1.z**2))
#print("l1_angular min: ", np.min(l1_angular))
#print("l1_angular max: ", np.max(l1_angular))
#print("")
#l2_angular = np.sqrt(l2.vx**2 + l2.vy**2 + l2.vz**2)/(np.sqrt(l2.x**2 + l2.y**2 + l2.z**2))
#print("l2_angular min: ", np.min(l2_angular))
#print("l2_angular max: ", np.max(l2_angular))

##print("")
##print("sun.ax:\t", sun.ax[::5000])
##print("sun.ay:\t", sun.ay[::5000])
##print("sun.az:\t", sun.az[::5000])
##print("")
##print("sun.vx:\t", sun.vx[::5000])
##print("sun.vy:\t", sun.vy[::5000])
##print("sun.vz:\t", sun.vz[::5000])
##print("")
##print("sun.x:\t", sun.x[::5000])
##print("sun.y:\t", sun.y[::5000])
##print("sun.z:\t", sun.z[::5000])

readyToLabel = True
plot1 = plt.figure(1)
for b in bodies:
	if b != earth and b != sun and b != l1:
		if not readyToLabel:
			plt.scatter(b.x, b.y, s = 1, c='red')
		else:
			plt.scatter(b.x, b.y, s = 1, c='red', label='Tracer particles')
			readyToLabel = False
plt.scatter(earth.x, earth.y, s=1, label='Earth')
plt.scatter(sun.x, sun.y, s=4, label='Sun')
#plt.scatter(l2.x, l2.y, s=1, label='l2')
plt.scatter(l1.x, l1.y, s=1, label='L1')
plt.legend(loc=1)

plot2 = plt.figure(2)
graph1 = plt.axes(projection = '3d')
for b in bodies:
	if b != earth and b != sun and b != l1:
		graph1.plot3D(b.x, b.y, b.z, c='gray')
		print("x: ", b.x[0] - l1.x[0], "\ty: ", b.y[0] - l1.y[0], " z: ", b.z[0])
graph1.plot3D(earth.x, earth.y, earth.z)
graph1.plot3D(sun.x, sun.y, sun.z)
graph1.plot3D(l1.x, l1.y, l1.z)

plot3 = plt.figure(3)
plt.plot(np.linspace(0.0, 1.0, 100), blocks[::50], '.r-')
plt.xlabel("Time (years)")
plt.ylabel("% of tracer particles that block sunlight")
plt.title("Blockage of sun rays due to tracer particles within a 1 km radius of L1")
#plt.legend(loc=1)

plt.savefig("C:/Users/khans/Desktop/UROP Spring 2022/plot.png")
plt.show()