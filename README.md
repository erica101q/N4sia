# Analytical Dynamics AE 544 Project
## Initial Plots
The intial plots are found in the zip file.
## Different Representations Analyzed
In this project, the representation of Euler angles, quaternions, and classical Rodrigues parameters were explored. The major problem with using Euler angles to analyze a system's rotation is gimbal lock. Gimbal lock occurs when the system loses a degree of freedom, which can cause inaccuracies. An ambiguity with quaternions is a negative sign change in the plots over time, but when the quaternion changes from positive to negative, it does not affect the attitude or angular velocity. Classical Rodrigues parameters have an ambiguity when phi in the scalar part of the equations is 180 degrees. When the system reaches this 180 degrees, it leads to inaccuracies and instability within the system. Each representation can show how a system behaves, but each system has its own faults. In the plots generated for Euler angles from the code, gimbal lock is detected when theta reaches or is close to pi/2. The plots also show the quaternion ambiguity with the negative sign flip. The classical Rodrigues parameters' ambiguity is seen in the plots when Bo in the plot is equal to zero utilizing the phi angle of 180 degrees. This is shown in the plots when the code is run to completion. 
![Rod_Plot_Final_AGA](https://github.com/user-attachments/assets/905bafb3-ce64-416e-80d3-182ec07f31af)
The plot above shows the classical Rodrigues plot and the time step is very small compared to the other plots. 
![Quat_Final_Plot](https://github.com/user-attachments/assets/849b70fd-41b3-494c-8911-23b2999aa0b2)
Another detection method with quaternions was converting them into another representation, such as Euler angles. Extending the time will allow for the quaternions to be converted into Euler angles and detect the gimbal lock. By running the code for a longer time, there will be more opportunity to detect the Euler angles from the quaternions. It can be observed that it takes a longer time to detect the gimbal lock within the quaternions because initially, the system of quaternions does not suffer from gimbal lock. However, when it is converted into another representation, such as Euler angles, gimbal lock is detected, which is another ambiguity in the system. In the code for testing the quaternion to euler angle representation, tspan should be about 200000 for detection of gimbal lock. Overall, using the graphs to show that the quaternions are not affected by gimbal lock and then turning it into euler angles is a great way to understand and analyse how the different representations behave.  
## Equations Used within the functions
<img width="392" alt="Euler_Angle_Screennshot_Equation" src="https://github.com/user-attachments/assets/4129a500-9e79-457f-9808-2cca14ef166a" />
<img width="1195" alt="Screen Shot 2025-03-12 at 1 56 14 PM" src="https://github.com/user-attachments/assets/19ef35cf-91b3-4297-adc1-1ef20585376d" />
<img width="1195" alt="Screen Shot 2025-03-12 at 2 00 34 PM" src="https://github.com/user-attachments/assets/393ce5b3-3aa2-4b96-a7e7-91a601da7b6a" />
<img width="1009" alt="Screen Shot 2025-03-12 at 2 01 42 PM" src="https://github.com/user-attachments/assets/934e31cb-7191-49f5-bf71-19ca5ddd6451" />
The next matrix is for classical Rodrigues Parameters. The book: Analytical Mechanics of Space Systems 2nd editon uses 'q' for Rodrigues Parameters. The following equations gives the kinematic differential equation for this particular representation
<img width="1008" alt="Screen Shot 2025-03-12 at 2 02 39 PM" src="https://github.com/user-attachments/assets/48343595-c3da-4e80-9df9-d17a3c3f1a37" />
For the angular velocity differential equations another book was used which can be found in the refrence [2]

<img width="624" alt="Screen Shot 2025-03-12 at 2 48 20 PM" src="https://github.com/user-attachments/assets/5be511c5-1df7-4474-9449-356329f601d0" />


## ODE 45, ODE 15, and ODE 23 Comparison
Each ODE function is a way to solve differential equations. However, ODE15s and ODE23s are for stiffer systems that can work well with nonlinear equations. What stiff ODE functions can do is smooth over the system, causing gimbal lock not to be detected, as seen in the code of euler angles. In the code, specifically for classical Rodrigues parameters, ODE23s was needed to solve for the differential equation because the Rodrigues parameters involved more complex and nonlinear equations. A major problem that arose when using ODE45 for classical Rodrigues parameters is that it didn't run for the entire time span selected, so a smaller time step was needed, and a different ODE function had to be used. Furthermore, a small change occurred when changing from ODE45 to ODE23s, as the system was able to extend for a second. However, because it was a small change, ODE45 and ODE15 were explored using the Euler angles. Moreover,if the time is put into increments the system can last up until the desired time of 200 seconds. This can be seen by changing the the following to: tspan=[0,10] then run the code, then change it again to tspan=[10,20]. Continue to run until 200 seconds. Furthermore, allowing the system to run without increments leads to failure with the ode45 function which is why another function was needed. However, all the systems worked when the time was reduced but ode23s was used because it was able to run for a second longer. However, as stated before the ode functions were more explored with the euler angles rather than the classical Rodrigues Parameters. The plot below shows the two functions in the Euler angles representation.
![Euler_ANgles_Final_Plots](https://github.com/user-attachments/assets/2735dcf9-1860-4ce5-985b-00f65d4e8161)

## Plot Explanation
The plot shows the system shifting towards the top away from the desired gimbal lock number which is pi/2. In the ode45 system the system is able to hit pi/2 and gimbal lock is detected however ode15s doesnt allow that. As stated before this is due to the way the differential equations are solved. Due to the function reducing the step size to a very small value the ambiguity is skipped over and the plot above shows that. Also in the code the detection for gimbal lock is not detected as well with ode15 function.  

## How to Use the MATLAB Code
### Notes for the code and Equations
In the code B is used for the classical Rodrigues parameters and q is used for quaternions. However, in the equations above q is used for the classical rodrigues parameters and B is used for the quaternions. This can be slightly confusing but it doesnt change anything. 

### Intial Values and What to Change 
* y0=[0.01,0.01,0.01,0.01,0.01,0.01]; For Euler Angles
* y0q=[1,0.01,0.01,0.01,0.01,0.01,0.01]; For Quaternions
* tspan=[0,2000];
* y0r=[1,0.01,0.01,0.01,0.01,0.01,0.01]; Rodrigues Parameters
* tspanr=[0,26];

The first 3 is the vector for euler angles and the first 4 is for the quaternions and rodrigues which is the scalar first and then the vector. The last 3 numbers for all are the angular velocites. wx,wy,wz or omega 1,omega 2, omega 3. 
These five variables are the variables to be changed within the code. However, the ambiguity and singularity are already set into place and the plots produced in the code are with the singularies. The above values are the intial conditions however in the code the variables change to the following:

* y0=[0.01,0.01,0.01,0.01,0.07,0.01]; For Euler Angles
* y0q=[1,0.01,0.01,0.01,0.01,0.07,0.01]; For Quaternions
* tspan=[0,2000];
* y0r=[1,0.01,0.01,0.01,0.01,0.07,0.01]; Rodrigues Parameters
* tspanr=[0,26];

Furthermore changing these variables will change the plots and detections of gimbal lock and the 180 degree error.
### Why change the wy value in the code?
In the assignment a hint was given to change the angular velocities. The angular velocitiy was changed until an ambiguity or singularity was displayed then that change was kept with all 3 representations. 
### What needs to be installed
MATLAB 2024a, ode45 tool,ode15s tool ,ode23s tool,
## Refrences and The use of AI
In the code, AI was used to help detect when the pitch angle reached π/2. This can be proven by looking at the graphs and observing when the second graph reached near 1.5 on the y-axis. AI was used to assist in making the animation logically correct. There were many errors within the animation, and detecting gimbal lock was another huge challenge that AI helped fix. The goal was to see when the marker showed up on the display, indicating that at least two lines had become aligned with each other, which would represent the gimbal lock problem.

Refrences:
 * [1] Schaub, H., & Junkins, J. L. (2009). Analytical mechanics of space systems (2nd edition.). American Institute of Aeronautics and Astronautics.
 * [2] Curtis, H. D. (2021). Orbital Mechanics for Engineering Students (Fourth edition.). Butterworth-Heinemann.

