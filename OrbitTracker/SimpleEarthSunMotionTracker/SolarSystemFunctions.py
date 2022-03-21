import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opt
from AstronomicalBody import Body


G = 6.67408e-8
N = 10000
earth_equ_radius = 637816000

## Finds the center of mass and velocity of a solar system ##
def centerMassAndVelocity(i, bodies):
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



### Tracks motion using leapfrog algorithm ###
def trackMotion(years, nsteps, bodies, earth, sun):
	tf = years * 2*np.pi*np.sqrt(np.sqrt((earth.x[0] - sun.x[0])**2 + (earth.y[0] - sun.y[0])**2 + (earth.z[0] - sun.z[0])**2)**3/G/sun.mass)
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



## Used as a function to minimize ##
## Finds the distance that l1 drifts from earth ##
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



## Used as a function to minimize ##
## Finds the distance that l1 drifts from earth ##
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
def trackMotionAndDrifts(years, nsteps, bodies, earth, sun, lagrangeBody, lagrangeGuess):
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

		# Find the difference in position between earth and l1 and earth and l2
		earthToLagrangeDrifts[i] = np.sqrt((lagrangeBody.x[i] - earth.x[i])**2 + (lagrangeBody.y[i] - earth.y[i])**2 + (lagrangeBody.z[i] - earth.z[i])**2) - lagrangeGuess

	return earthToLagrangeDrifts



## A more specific version of trackMotion which also tracks whether or not the object at the lagrange point
## blocks a ray of light from the center of the sun to the earth. Returns an array of bits. 0 if the lagrange
## point did not block the ray of light. 1 if the lagrange point did block the ray of light
########## NOT SURE IF THIS WILL WORK IN 3D ##########
def trackMotionAndRays(years, nsteps, bodies, earth, sun, lagrangeBody, lagrangeGuess):
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

		# Find whether or not the object at the lagrange point blocked a ray from the sun to the earth
		#distance_from_lagrange_to_sun = np.sqrt((lagrangeBody.x[i] - sun.x[i])**2 + (lagrangeBody.y[i] - sun.y[i])**2 + (lagrangeBody.z[i] - sun.z[i])**2)
		distance_from_earth_to_sun = np.sqrt((earth.x[i] - sun.x[i])**2 + (earth.y[i] - sun.y[i])**2 + (earth.z[i] - sun.z[i])**2)
		distance_from_earth_to_l1 = np.sqrt((earth.x[i] - lagrangeBody.x[i])**2 + (earth.y[i] - lagrangeBody.y[i])**2 + (earth.z[i] - lagrangeBody.z[i])**2)

		r_l_hatX = (lagrangeBody.x[i] - earth.x[i]) / distance_from_earth_to_l1
		r_l_hatY = (lagrangeBody.y[i] - earth.y[i]) / distance_from_earth_to_l1
		r_l_hatZ = (lagrangeBody.z[i] - earth.z[i]) / distance_from_earth_to_l1

		r_s_hatX = (sun.x[i] - earth.x[i]) / distance_from_earth_to_sun
		r_s_hatY = (sun.y[i] - earth.y[i]) / distance_from_earth_to_sun
		r_s_hatZ = (sun.z[i] - earth.z[i]) / distance_from_earth_to_sun

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



## Tracks the average distance that l1 drifts from earth ##
def mean_l1_drifts_from_earth(l1Start):
	earth = Body(M[0], N, X[0], Y[0], Z[0], VX[0], vy_init, VZ[0])
	sun = Body(M[1], N, X[1], Y[1], Z[1], VX[1], VY[1], VZ[1])

	r_es = np.sqrt((sun.x[0] - earth.x[0])**2 + (sun.y[0] - earth.y[0])**2 + (sun.z[0] - earth.z[0])**2)

	l1_vy = l1Start * np.sqrt(G*sun.mass/r_es**3)
	l1 = Body(0, N, l1Start, 0, 0, 0, l1_vy, 0)

	bodies = np.array([earth, sun, l1])
	earthToL1Drifts = trackMotionAndDrifts(Years, N, bodies, l1, l1Start)
	return np.mean(earthToL1Drifts)



## Tracks the average distance that l2 drifts from earth ##
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