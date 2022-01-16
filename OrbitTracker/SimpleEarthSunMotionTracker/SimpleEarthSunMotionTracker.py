import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

G = 6.67408e-8

def trackEarthMotion(numberOfYears, numberOfSteps):
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
	vy[0] = np.sqrt(G*M[1]/x[0])
	print("vy[0]: ", vy[0])

	tf = numberOfYears * 2*np.pi*np.sqrt(x[0]**3/G/M[1])
	dt = tf / numberOfSteps

	# Acceleration should be negative if the earth's position is greater than the sun's position,
	# positive if the earth's position is less than the sun's position	
	r = np.sqrt(x[0]**2 + y[0]**2 + z[0]**2)
	ax[0] =  -G*M[1]/r**2 * x[0]/r
	ay[0] =  -G*M[1]/r**2 * y[0]/r
	az[0] =  -G*M[1]/r**2 * z[0]/r

	for i in range(1, numberOfSteps):
		x_temp = x[i-1] + vx[i-1]*0.5*dt
		y_temp = y[i-1] + vy[i-1]*0.5*dt
		z_temp = z[i-1] + vz[i-1]*0.5*dt

		r = np.sqrt(x_temp**2 + y_temp**2 + z_temp**2)
		acc = G*M[1]/r**2;

		# Acceleration is recorded at t = 0, 0.5dt, 1.5dt, 2.5dt, ...
		ax[i] = acc * -x_temp/r
		ay[i] = acc * -y_temp/r
		az[i] = acc * -z_temp/r

		vx[i] = vx[i-1] + ax[i]*dt
		vy[i] = vy[i-1] + ay[i]*dt
		vz[i] = vz[i-1] + az[i]*dt

		x[i] = x_temp + vx[i]*0.5*dt
		y[i] = y_temp + vy[i]*0.5*dt
		z[i] = z_temp + vz[i]*0.5*dt

	return x, y, z, vx, vy, vz, ax, ay, az

x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr, ax_arr, ay_arr, az_arr = trackEarthMotion(1, 100000)
#print("***********X Position**********")
#print(x_arr)
#print("***********X Velocity**********")
#print(vx_arr)
#print("***********X Acceleration**********")
#print(ax_arr)	    

#print("***********Y Position**********")
#print(y_arr)	
#print("***********Y Velocity**********")
#print(vy_arr)	    
#print("***********Y Acceleration**********")
#print(ay_arr)

plot1 = plt.figure(1)
plt.scatter(x_arr, y_arr, s=1)

plot2 = plt.figure(2)
graph1 = plt.axes(projection = '3d')
graph1.plot3D(x_arr, y_arr, z_arr)

print("x initial: ", x_arr[0])
print("x final: ", x_arr[-1])
print("y initial: ", y_arr[0])
print("y final: ", y_arr[-1])

plt.show()


