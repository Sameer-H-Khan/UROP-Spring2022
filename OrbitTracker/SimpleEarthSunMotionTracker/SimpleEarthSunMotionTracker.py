import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

G = 6.67408e-8

def trackEarthMotion(numberOfYears, numberOfSteps):
	tf = numberOfYears * 31557600.0		# Converting years to seconds. Using 365.25 days/year. May be slightly inaccurate
	dt = tf / numberOfSteps

	# Values for earth
	x = np.zeros(numberOfSteps)
	y = np.zeros(numberOfSteps)
	z = np.zeros(numberOfSteps)
	vx = np.zeros(numberOfSteps)
	vy = np.zeros(numberOfSteps)
	vz = np.zeros(numberOfSteps)
	ax = np.zeros(numberOfSteps)
	ay = np.zeros(numberOfSteps)
	az = np.zeros(numberOfSteps)

	x[0] = X[0]; y[0] = Y[0]; z[0] = Z[0]; vx[0] = VX[0]; vy[0] = VY[0]; vz[0] = VZ[0];

	# Acceleration should be negative if the earth's position is greater than the sun's position,
	# positive if the earth's position is less than the sun's position
	ax[0] = G*M[1]/((X[1] - x[0])*np.abs(X[1] - x[0]))	

	# Because both the earth and the sun start out at y, z = 0, 0, finding the acceleration in these
	# directions is weird because r = 0, so this will be ignored for now
	#ay[0] = G*M[1]/((Y[1] - y[0])*np.abs(Y[1] - y[0]))		 
	#az[0] = G*M[1]/((Z[1] - z[0])*np.abs(Z[1] - z[0]))

	for i in range(1, numberOfSteps):
		x[i] = x[i-1] + vx[i-1]*dt
		y[i] = y[i-1] + vy[i-1]*dt
		z[i] = z[i-1] + vz[i-1]*dt;

		vx[i] = vx[i-1] + ax[i-1]*dt
		vy[i] = vy[i-1] + ay[i-1]*dt
		vz[i] = vz[i-1] + az[i-1]*dt;

		#if(X[1] - x[i] == 0):
		#	ax[i] = 0
		#else:
		ax[i] = G*M[1]/((X[1] - x[i])*np.abs(X[1] - x[i]))		

		#if(Y[1] - y[i] == 0):
		#	ay[i] = 0
		#else:
		ay[i] = G*M[1]/((Y[1] - y[i])*np.abs(Y[1] - y[i]))

		# I'm ignoring the z-component for now, just trying to get it to work in 2 dimensions first
		#if(Z[1] - z[i] == 0):
		#	az[i] = 0
		#else:
		#az[i] = G*M[1]/((Z[1] - z[i])*np.abs(Z[i] - z[i]))

	return x, y, z, vx, vy, vz, ax, ay, az

x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, ax_arr, ay_arr, az_arr = trackEarthMotion(0.0001, 100)
print("***********X Position**********")
print(x_arr)	    
print("***********Y Position**********")
print(y_arr)	
print("***********Y Velocity**********")
print(vy_arr)	    
print("***********Y Acceleration**********")
print(ay_arr)
plt.scatter(x_arr, y_arr)
plt.show()


