%Analytical Dynamics Project
clc
clear
close all
fprintf('This program does not start with the intial plots. It starts with the singularity\nor ambigutiy added to them.\nSo the angular velocities are already manuipulated.\nTo see the original plots follow directions in the README.\n\n')
fprintf('In order to run this program enough time needs to be given. \nThe intial plots should be set as the READMe explains. \nThe original plots should be given a time of 2000 seconds to run. \n\n')
%Can Assign arbitary values to my Inertia to solve my differential equations.
I1=1;I2=2;I3=3;M1=0;M2=0;M3=0;
fprintf('To get the maniuplated plots input changes in angular velocities which is just the wy component into the\nintial arrays labeled y0,y0q, and y0r.\nThis is also given in the READMe.\n')
fprintf('For the intial plot there should be no gimbal lock detected.\nHowever, when the angular velocites are manuipulated the statement will change.\n\n')

%!!!!!!!Manipulate these variables within the code to see any changes!!

%Each array has either the angles,quaternions, or rodrigues parameters
%first then followed by the 3 components of angular velocity.

%y0 is euler angles
y0=[0.01,0.01,0.01,0.01,0.07,0.01];
%y0q are the quaternions
y0q=[1,0.01,0.01,0.01,0.01,0.07,0.01];
tspan=[0,2000];
%y0r is the Rodrigues Parameters and time span. There is a different tspan
%becuase the ode function can't handle the larger time span
y0r=[1,0.01,0.01,0.01,0.01,0.07,0.01];
tspanr=[0,26];

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%The animation is for when there is a gimbal lock is detected.

%Keeping the euler angles the first values and the angular velocity the
%next 3 terms.  pi/2)*2.5708 is as close as I can get it to be close to pi/2 for the
%pitch resulting in gimbal lock
%For the intial conditions of y0=[0.01, 0.01, 0.01,0.01,0.01,0.01]  gimbal lock happens when wy starts at 0.07
%y0=[0.01,0.01,0.01,0.01,0.07,0.01];  %4.8, 4.8, 4.8]; %Inital Angular Velocity
%tspan=[0,2000]; %Time 200
[t,Angles]=ode45(@(t,y)(odefun(t,y,I1,I2,I3)),tspan,y0);
%Can have a smoothing affect the other solvers
%[t,Angles]=ode23s(@(t,y)(odefun(t,y,I1,I2,I3)),tspan,y0);
fprintf('Gimbal Lock will not be detected for the quaternions converting to the euler angles if\n enough time is not given.\n')
gimbal_lock_detected = any(abs(Angles(:, 2)) > pi/2 - 1e-3);
if gimbal_lock_detected
    lock_time = t(find(abs(Angles(:, 2)) > pi/2 - 1e-3, 1));  % Find the first time gimbal lock happens
    fprintf('For the Euler Angles the gimbal lock is detected at time t = %.2f seconds\n', lock_time);
else
    disp('No gimbal lock detected.');
end
%Plotting
figure ()
subplot(6,1,1);plot(t,Angles(:,1));title('Psi vs time');xlabel('time(seconds)');ylabel('psi angle')
hold off
subplot(6,1,2);
plot(t,Angles(:,2));
title('theta vs time');
xlabel('time(seconds)');
ylabel('Theta or Pitch Angle');
hold off
subplot(6,1,3);plot(t,Angles(:,3));title('phi vs Time');xlabel('time(seconds)');ylabel('Phi angle');
hold off
subplot(6,1,4);plot(t,Angles(:,4));title('Omega 1 vs Time');xlabel('time(seconds)');ylabel('Omega 1')
hold off
subplot(6,1,5);plot(t,Angles(:,5));title('Omega 2 vs Time');xlabel('time(seconds)');ylabel('Omega 2')
hold off
subplot(6,1,6);plot(t,Angles(:,6));title('Omega 3 vs Time');xlabel('time(seconds)');ylabel('Omega 3')
hold off
[t_ode45, Angles_ode45] = ode45(@(t, y) odefun(t, y, I1, I2, I3), tspan, y0);
% Solve using ode15s
[t_ode15s, Angles_ode15s] = ode15s(@(t, y) odefun(t, y, I1, I2, I3), tspan, y0);
% Check for gimbal lock detection in both solvers
gimbal_lock_ode45 = any(abs(Angles_ode45(:, 2)) > pi/2 - 1e-3);
gimbal_lock_ode15s = any(abs(Angles_ode15s(:, 2)) > pi/2 - 1e-3);
% Print gimbal lock detection results
if gimbal_lock_ode45
    lock_time_ode45 = t_ode45(find(abs(Angles_ode45(:, 2)) > pi/2 - 1e-3, 1));  % Find the first time gimbal lock happens for ode45
    fprintf('For ode45, gimbal lock is detected at time t = %.2f seconds\n', lock_time_ode45);
else
    disp('No gimbal lock detected with ode45.');
end
if gimbal_lock_ode15s
    lock_time_ode15s = t_ode15s(find(abs(Angles_ode15s(:, 2)) > pi/2 - 1e-3, 1));  % Find the first time gimbal lock happens for ode15s
    fprintf('For ode15s, gimbal lock is detected at time t = %.2f seconds\n', lock_time_ode15s);
else
    disp('No gimbal lock detected with ode15s.');
end
% Plotting both results
figure;
% Plot Psi (Roll) for both solvers
subplot(6, 1, 1);
plot(t_ode45, Angles_ode45(:, 1), 'r', t_ode15s, Angles_ode15s(:, 1), 'b');
title('Psi vs Time');
xlabel('Time (seconds)');
ylabel('Psi (Roll)');
legend('ode45', 'ode15s');
% Plot Theta (Pitch) for both solvers
subplot(6, 1, 2);
plot(t_ode45, Angles_ode45(:, 2), 'r', t_ode15s, Angles_ode15s(:, 2), 'b');
title('Theta vs Time');
xlabel('Time (seconds)');
ylabel('Theta (Pitch)');
legend('ode45', 'ode15s');
% Plot Phi (Yaw) for both solvers
subplot(6, 1, 3);
plot(t_ode45, Angles_ode45(:, 3), 'r', t_ode15s, Angles_ode15s(:, 3), 'b');
title('Phi vs Time');
xlabel('Time (seconds)');
ylabel('Phi (Yaw)');
legend('ode45', 'ode15s');
% Plot Omega 1 for both solvers
subplot(6, 1, 4);
plot(t_ode45, Angles_ode45(:, 4), 'r', t_ode15s, Angles_ode15s(:, 4), 'b');
title('Omega 1 vs Time');
xlabel('Time (seconds)');
ylabel('Omega 1');
legend('ode45', 'ode15s');
% Plot Omega 2 for both solvers
subplot(6, 1, 5);
plot(t_ode45, Angles_ode45(:, 5), 'r', t_ode15s, Angles_ode15s(:, 5), 'b');
title('Omega 2 vs Time');
xlabel('Time (seconds)');
ylabel('Omega 2');
legend('ode45', 'ode15s');
% Plot Omega 3 for both solvers
subplot(6, 1, 6);
plot(t_ode45, Angles_ode45(:, 6), 'r', t_ode15s, Angles_ode15s(:, 6), 'b');
title('Omega 3 vs Time');
xlabel('Time (seconds)');
ylabel('Omega 3');
legend('ode45', 'ode15s');
%Quaterion and plotting 
%Gimbal lock happens when the angular velocity (wy) is 2
%Takes a longer to detect with the quaternions 1st one is a scalar
%y0q=[1,0.01,0.01,0.01,0.01,0.07,0.01];
[t,Quaterions]=ode45(@(t,Matrix_of_7)(ThedifferentialEquation1(t,Matrix_of_7,I1,I2,I3)),tspan,y0q);
euler_angles = zeros(length(t), 3);
for i = 1:length(t)
    q = Quaterions(i, 1:4);
    q = q / norm(q);
    R = quaternion_to_rotation_matrix(q);
    [roll, pitch, yaw] = rotation_matrix_to_euler(R);
    euler_angles(i, :) = [roll, pitch, yaw];
end
figure ()
plot(t,euler_angles(:,2))
title('Quaternion to Euler Angle (Pitch)')
xlabel('Time(Seconds)')
ylabel('Pitch angle')
hold off
gimbal_lock_detectedag = any(abs(euler_angles(:, 2)) > pi/2 - 1e-3);
if gimbal_lock_detectedag
    lock_timeag = t(find(abs(euler_angles(:, 2)) > pi/2 - 1e-3, 1));  % Find the first time gimbal lock happens
    fprintf('For quaternions to euler angles the gimbal lock is detected at time t = %.2f seconds\n. Gimbal lock is detected later because the system is in a quaternion state at first \n which doesnt suffer from gimbal lock \n. The system does not detect gimbal lock until the conversion \n process to euler angles is done \n.', lock_timeag);
else
    disp('No gimbal lock detected.');
end
%Gimbal lock is detected later because the system is in a quaternion state
%at first and because quaternions are not affected by gimbal lock the
%system does not detect it until the conversion process is done. 
figure ()
%Plottiing
subplot(7,1,1);plot(t,Quaterions(:,1));title('Quaterion 4 vs time');xlabel('time(seconds)');ylabel('Quaterion 4')
hold off
subplot(7,1,2);
plot(t,Quaterions(:,2));
title('Quaterion 1 vs time');
xlabel('time(seconds)');
ylabel('Quaterion 1');
hold off
subplot(7,1,3);plot(t,Quaterions(:,3));title('Quaterion 2 vs Time');xlabel('time(seconds)');ylabel('Quaterion 2');
hold off
subplot(7,1,4);plot(t,Quaterions(:,4));title('Quaterion 3 vs Time');xlabel('time(seconds)');ylabel('Quaterion 3');
hold off
subplot(7,1,5);plot(t,Quaterions(:,5));title('Omega 1 vs Time');xlabel('time(seconds)');ylabel('Omega 1');
hold off
subplot(7,1,6);plot(t,Quaterions(:,6));title('Omega 2 vs Time');xlabel('time(seconds)');ylabel('Omega 2');
hold off
subplot(7,1,7);plot(t,Quaterions(:,7));title('Omega 3 vs Time');xlabel('time(seconds)');ylabel('Omega 3');
%Rod Parameters
%y0r=[1,0.01,0.01,0.01,0.01,0.07,0.01];
%tspanr=[0,26];
%[t,RodPar]=ode15s(@(t,quat)(oderodfun(t,quat,I1,I2,I3)),tspanr,y0r);
%[t,RodPar]=ode45(@(t,quat)(oderodfun(t,quat,I1,I2,I3)),tspanr,y0r);
[t,RodPar]=ode23s(@(t,quat)(oderodfun(t,quat,I1,I2,I3)),tspanr,y0r);
hold off
figure()
subplot(7,1,1);
plot(t,RodPar(:,1));title('Rodrigues Scalar(Bo) vs Time');xlabel('time(seconds)');ylabel('Bo');subplot(7,1,2);
plot(t,RodPar(:,2));title('Rodrigues vector 1(B1) vs Time');xlabel('time(seconds)');ylabel('B1');subplot(7,1,3);
plot(t,RodPar(:,3));title('Rodrigues vector 2 (B2)');xlabel('time(seconds)');ylabel('B2');subplot(7,1,4);
plot(t,RodPar(:,4));title('Rodrigues vector 3 (B3)');xlabel('time(seconds)');ylabel('B3');subplot(7,1,5);
plot(t,RodPar(:,5));title('Omega 1 vs Time');xlabel('time(seconds)');ylabel('Omega 1');subplot(7,1,5);
subplot(7,1,6);plot(t,RodPar(:,6));title('Omega 2 vs Time');xlabel('time(seconds)');ylabel('Omega 2');
subplot(7,1,7);plot(t,RodPar(:,7));title('Omega 3 vs Time');xlabel('time(seconds)');ylabel('Omega 3');
fprintf('\n\nDue to the nonlinear equations of Rodrigues Parameters ode45 is unable to solve\n with a larger time step. Increasing the time will lead to a warning and inaccuracy.\n')
%Using ode15 for stiffness and finding the 180 degree error in Rodrigues
%Parameter
threshold = 1e-1;  % Threshold to detect when Bo is close to 0
gimbalLockTime = NaN;  % Initialize gimbal lock time
for i = 1:length(RodPar(:,1))
    if abs(RodPar(i, 1)) < threshold
        gimbalLockTime = t(i); 
        break;  
    end
end
if ~isnan(gimbalLockTime)
    fprintf('For Bo: phi=180 ambiguity detected at time t = %.2f seconds\n', gimbalLockTime);
else 
    fprintf('For Bo: No phi=180 ambiguity detected during the simulation.\n');
end
% 3D Animation with Euler Angles showing Gimbal Lock (Smooth Transition)
theta = linspace(0, 2*pi, 100); 
circle_x = cos(theta);
circle_y = sin(theta);
circle_z = zeros(size(theta)); 
circle2_x = zeros(size(theta));  
circle2_y = cos(theta);  
circle2_z = sin(theta);  
circle3_x = cos(theta);  
circle3_y = zeros(size(theta));  
circle3_z = sin(theta);  
figure;
axis equal;
title('3D Animation of Gimbal Lock');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on;
h1 = plot3(circle_x, circle_y, circle_z, 'g'); 
h2 = plot3(circle2_x, circle2_y, circle2_z, 'b');
h3 = plot3(circle3_x, circle3_y, circle3_z, 'm');
axis([-1 1 -1 1 -1 1]);
gif_filename = 'gimbal_lock_animation.gif';
t = linspace(0, 4*pi, 300);  
h_marker = plot3(nan, nan, nan, 'ko', 'MarkerFaceColor', 'k'); 
tolerance = 1; 
for i = 1:length(t)
    roll = 90 * sin(t(i)); 
    pitch = 90 * cos(t(i));  
    yaw = 0;   
    R = eul2rotm([yaw, pitch, roll], 'ZYX'); 
    rotated_circle1 = R * [circle_x; circle_y; circle_z];
    rotated_circle2 = R * [circle2_x; circle2_y; circle2_z];
    rotated_circle3 = R * [circle3_x; circle3_y; circle3_z];
    set(h1, 'XData', rotated_circle1(1, :), 'YData', rotated_circle1(2, :), 'ZData', rotated_circle1(3, :));
    set(h2, 'XData', rotated_circle2(1, :), 'YData', rotated_circle2(2, :), 'ZData', rotated_circle2(3, :));
    set(h3, 'XData', rotated_circle3(1, :), 'YData', rotated_circle3(2, :), 'ZData', rotated_circle3(3, :));
    if abs(abs(pitch) - 90) < tolerance  
        R1 = R * [circle_x; circle_y; circle_z];
        R2 = R * [circle2_x; circle2_y; circle2_z];
        R3 = R * [circle3_x; circle3_y; circle3_z];
        dot_product_12 = dot(R1(:,1), R2(:,1));  
        dot_product_13 = dot(R1(:,1), R3(:,1));  
        angle_12 = acos(dot_product_12) * (180/pi);
        angle_13 = acos(dot_product_13) * (180/pi);
        if angle_12 < tolerance || angle_13 < tolerance
            set(h_marker, 'XData', 0, 'YData', 0, 'ZData', 0); 
            text(0.1, 0.1, 0.1, 'Gimbal Lock', 'Color', 'black', 'FontSize', 12);
        else
            set(h_marker, 'XData', nan, 'YData', nan, 'ZData', nan);
        end
    else
        set(h_marker, 'XData', nan, 'YData', nan, 'ZData', nan);
    end
    frame = getframe(gcf);
    [A, map] = rgb2ind(frame.cdata, 256);
    if i == 1
        imwrite(A, map, gif_filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        imwrite(A, map, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
    pause(0.1); 
end
figure;
for i = 1:length(t)
    img = imread('gimbal_lock_animation.gif', 'Frames', i);
    imshow(img);
    pause(0.01); % Adjust speed if needed
end
options = odeset('RelTol', 1e-3, 'AbsTol', 1e-4, 'MaxStep', 0.1);
%[t,RodParstif]=ode15i(@(t,quat)(oderodfun(t,quat,I1,I2,I3)),tspanr,y0r,options);
function dr_dt=oderodfun(t,quat,I1,I2,I3)
M1=0;M2=0;M3=0;%I1=1;I2=2;I3=3;
        dr_dt=zeros(7,1);dr_dt(1)=1/2*((-quat(1)*0)-quat(2)*quat(5)-quat(3)*quat(6)-quat(4)*quat(7));%q0
        dr_dt(2)=1/2*((1+quat(2).^2)*quat(5) + ((quat(2)*quat(3)-quat(4))*quat(6)) + (quat(2)*quat(4)+quat(3))*quat(7));%q1
        dr_dt(3)=1/2*((quat(3)*quat(2) + quat(4))*quat(5) + (1+quat(3).^2)*quat(6) + (quat(3)*quat(4)-quat(2))*quat(7));%q2
        dr_dt(4)=1/2*((quat(4)*quat(2)-quat(3))*quat(5) + quat(4)*quat(3) + quat(2)*quat(6) + (1+quat(4).^2)*quat(7));%q3
        dr_dt(5)=(M1-(I3-I2)*(quat(6))*(quat(7)))/I1;
        dr_dt(6)=(M2-(I1-I3)*(quat(7))*(quat(5)))/I2;
        dr_dt(7)=(M3-(I2-I1)*(quat(5))*(quat(6)))/I3;
        %Normalizing 
        normRodrigues = norm(quat(2:4));  
    if normRodrigues ~= 0
        quat(2:4) = quat(2:4) / normRodrigues;  % Normalize the Rodrigues vector
    end
end
%Euler Angles
function da_dt=odefun(t,y,I1,I2,I3)
M1=0;M2=0;M3=0;
    da_dt=zeros(6,1);%omega y= y(5), omega z=y(6)
        da_dt(1) = y(4) + tan(y(2)) * (y(5) * sin(y(1)) + y(6) * cos(y(1)));
        da_dt(2) = y(5) * cos(y(1)) - y(6) * sin(y(1));
        da_dt(3) = (sin(y(1))/cos(y(2))) * y(5) + (cos(y(1))/cos(y(2))) * y(6);
        %Angular velocities
        da_dt(4)=(M1-(I3-I2)*(y(5))*(y(6)))/I1;
        da_dt(5)=(M2-(I1-I3)*(y(6))*(y(4)))/I2;
        da_dt(6)=(M3-(I2-I1)*(y(4))*(y(5)))/I3;
end
%Quaterion
%Turning from quat to euler angles
function [roll, pitch, yaw] = rotation_matrix_to_euler(R)
    % Roll (phi)
    roll = atan2(R(3, 2), R(3, 3));
    % Pitch (theta)
    pitch = atan2(-R(3, 1), sqrt(R(3, 2)^2 + R(3, 3)^2));
    % Yaw (psi)
    yaw = atan2(R(2, 1), R(1, 1));
end
function R = quaternion_to_rotation_matrix(q)
    % Extract the components of the quaternion
    q0 = q(1);
    q1 = q(2);
    q2 = q(3);
    q3 = q(4);
    R = [
        1 - 2*(q2^2 + q3^2),  2*(q1*q2 - q0*q3),  2*(q1*q3 + q0*q2);
        2*(q1*q2 + q0*q3),  1 - 2*(q1^2 + q3^2),  2*(q2*q3 - q0*q1);
        2*(q1*q3 - q0*q2),  2*(q2*q3 + q0*q1),  1 - 2*(q1^2 + q2^2)
    ];
end
function y = ThedifferentialEquation1(t, Matrix_of_7,I1,I2,I3)
q = Matrix_of_7(1:4);  % Quaternion
omega = Matrix_of_7(5:7); q = q / norm(q);
% Constants
    mgx = 0; 
    mgy = 0;  
    mgz = 0; 
    y1 = 0.5 * quaternionMatrix(q, [omega; 0]);
    y2 = [
        (mgx - (I3 - I2) * omega(2) * omega(3)/ I1);
        (mgy - (I1 - I3) * omega(1) * omega(3)/ I2);
        (mgz - (I2 - I1) * omega(1) * omega(2)/ I3);
    ];    
    y = [y1; y2];
end
function quaternionresults = quaternionMatrix(q1, q2)
    quaternionresults = [
        q1(4) * q2(1) + q1(1) * q2(4) + q1(2) * q2(3) - q1(3) * q2(2);
        q1(4) * q2(2) + q1(2) * q2(4) + q1(3) * q2(1) - q1(1) * q2(3);
        q1(4) * q2(3) + q1(3) * q2(4) + q1(1) * q2(2) - q1(2) * q2(1);
        q1(4) * q2(4) - q1(1) * q2(1) - q1(2) * q2(2) - q1(3) * q2(3);
    ];
end
