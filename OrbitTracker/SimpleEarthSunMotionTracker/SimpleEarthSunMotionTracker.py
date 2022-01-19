import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# The following values are for mass, position components, and velocity components
# Values for earth are at index 0, values for sun are at index 1
M, X, Y, Z, VX, VY, VZ = np.loadtxt("InitialConditions.txt", unpack = True)

G = 6.67408e-8

def trackEarthMotion(numberOfYears, numberOfSteps):
	# Values for earth
	x_e = np.zeros(numberOfSteps)
	y_e = np.zeros(numberOfSteps)
	z_e = np.zeros(numberOfSteps)
	vx_e = np.zeros(numberOfSteps)
	vy_e = np.zeros(numberOfSteps)
	vz_e = np.zeros(numberOfSteps)
	ax_e = np.zeros(numberOfSteps)
	ay_e = np.zeros(numberOfSteps)
	az_e = np.zeros(numberOfSteps)

	# Values for sun
	x_s = np.zeros(numberOfSteps)
	y_s = np.zeros(numberOfSteps)
	z_s = np.zeros(numberOfSteps)
	vx_s = np.zeros(numberOfSteps)
	vy_s = np.zeros(numberOfSteps)
	vz_s = np.zeros(numberOfSteps)
	ax_s = np.zeros(numberOfSteps)
	ay_s = np.zeros(numberOfSteps)
	az_s = np.zeros(numberOfSteps)

	x_e[0] = X[0]; y_e[0] = Y[0]; z_e[0] = Z[0]; vx_e[0] = VX[0]; vy_e[0] = VY[0]; vz_e[0] = VZ[0];
	vy_e[0] = np.sqrt(G*M[1]/x_e[0])
	print("vy[0]: ", vy_e[0])

	x_s[0] = X[1]; y_s[0] = Y[1]; z_s[0] = Z[1]; vx_s[0] = VX[1]; vy_s[0] = VY[1]; vz_s[0] = VZ[1];

	tf = numberOfYears * 2*np.pi*np.sqrt(x_e[0]**3/G/M[1])
	dt = tf / numberOfSteps

	# Earth acceleration should be negative if the earth's position is greater than the sun's position,
	# positive if the earth's position is less than the sun's position	
	r_e = np.sqrt(x_e[0]**2 + y_e[0]**2 + z_e[0]**2)
	ax_e[0] =  -G*M[1]/r_e**2 * x_e[0]/r_e
	ay_e[0] =  -G*M[1]/r_e**2 * y_e[0]/r_e
	az_e[0] =  -G*M[1]/r_e**2 * z_e[0]/r_e

	ax_s[0] =  G*M[0]/r_e**2 * x_e[0]/r_e
	ay_s[0] =  G*M[0]/r_e**2 * y_e[0]/r_e
	az_s[0] =  G*M[0]/r_e**2 * z_e[0]/r_e

	for i in range(1, numberOfSteps):
		# Calculating position at half step
		x_e_temp = x_e[i-1] + vx_e[i-1]*0.5*dt
		y_e_temp = y_e[i-1] + vy_e[i-1]*0.5*dt
		z_e_temp = z_e[i-1] + vz_e[i-1]*0.5*dt

		x_s_temp = x_s[i-1] + vx_s[i-1]*0.5*dt
		y_s_temp = y_s[i-1] + vy_s[i-1]*0.5*dt
		z_s_temp = z_s[i-1] + vz_s[i-1]*0.5*dt

		# Calculating acceleration at half step
		r_e = np.sqrt((x_e_temp - x_s_temp)**2 + (y_e_temp - y_s_temp)**2 + (z_e_temp - z_s_temp)**2)
		acc_e = G*M[1]/r_e**2;

		r_s = np.sqrt((x_s_temp - x_e_temp)**2 + (y_s_temp - y_e_temp)**2 + (z_s_temp - z_e_temp)**2)
		acc_s = G*M[0]/r_s**2;

		# Calculating component acceleration at half step. Acceleration is recorded at t = 0, 0.5dt, 1.5dt, 2.5dt, ...
		ax_e[i] = acc_e * -x_e_temp/r_e
		ay_e[i] = acc_e * -y_e_temp/r_e
		az_e[i] = acc_e * -z_e_temp/r_e

		ax_s[i] = acc_s * -x_s_temp/r_s
		ay_s[i] = acc_s * -y_s_temp/r_s
		az_s[i] = acc_s * -z_s_temp/r_s

		# Calculating velocity
		vx_e[i] = vx_e[i-1] + ax_e[i]*dt
		vy_e[i] = vy_e[i-1] + ay_e[i]*dt
		vz_e[i] = vz_e[i-1] + az_e[i]*dt

		vx_s[i] = vx_s[i-1] + ax_s[i]*dt
		vy_s[i] = vy_s[i-1] + ay_s[i]*dt
		vz_s[i] = vz_s[i-1] + az_s[i]*dt

		# Calculating position at full step
		x_e[i] = x_e_temp + vx_e[i]*0.5*dt
		y_e[i] = y_e_temp + vy_e[i]*0.5*dt
		z_e[i] = z_e_temp + vz_e[i]*0.5*dt

		x_s[i] = x_s_temp + vx_s[i]*0.5*dt
		y_s[i] = y_s_temp + vy_s[i]*0.5*dt
		z_s[i] = z_s_temp + vz_s[i]*0.5*dt

	return x_e, y_e, z_e, vx_e, vy_e, vz_e, ax_e, ay_e, az_e, x_s, y_s, z_s, vx_s, vy_s, vz_s, ax_s, ay_s, az_s

x_e_arr, y_e_arr, z_e_arr, vx_e_arr, vy_e_arr, vz_e_arr, ax_e_arr, ay_e_arr, az_e_arr, x_s_arr, y_s_arr, z_s_arr, vx_s_arr, vy_s_arr, vz_s_arr, ax_s_arr, ay_s_arr, az_s_arr = trackEarthMotion(10, 100000)

plot1 = plt.figure(1)
plt.scatter(x_e_arr, y_e_arr, s=1)
plt.scatter(x_s_arr, y_s_arr, s=1)

plot2 = plt.figure(2)
graph1 = plt.axes(projection = '3d')
graph1.plot3D(x_e_arr, y_e_arr, z_e_arr)
graph1.plot3D(x_s_arr, y_s_arr, z_s_arr)

print("x initial: ", x_e_arr[0])
print("x final: ", x_e_arr[-1])
print("y initial: ", y_e_arr[0])
print("y final: ", y_e_arr[-1])

plt.show()


