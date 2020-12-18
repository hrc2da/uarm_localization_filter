%% Setup
clc,clear
close all
rng(100)  %this creates repeatable random numbers

%% System

names = ['x' 'y' 'z']; %state names to be used for plots
% Discrete time system matrices
F = eye(3);
B = eye(3);
H = [1 0 0;
     0 1 0];
D = 0;
system = ss(F, B, H, D, 1);
% define sample time as 1 second for simplicity

x0 = [0; 150; 150]; % default starting position of uArm
t = 0:10; %time steps
n_k = length(t); % number of time steps

% Process noise
w_stdev = 12;  %standard deviation of process noise
Q = w_stdev^2;

% Sensor
R_noise = [10^2 0; % guess for variance on noise for camera measurement
           0 10^2];

%% Simulation

x_desired = [200; 200; 200]; % desired location

% Tuning parameters
R_filter = [10^2 0; % guess for variance on noise for camera measurement
           0 10^2]; %much less confident in z measurement
Q_filter = 12^2;
       
G = eye(3); % process noise is simply additive
P0 = [100^2 0 0;
      0 100^2 0;
      0 0 200^2];

Pgate = 0.90;
Lam0 = chi2inv(Pgate, 2);   %statistic threshold
rej=0;  % initialize number of rejected values

%initialize simulation values
xtrue=zeros(3,n_k);
xtrue(:,1)=x0;
xhat = xtrue;
P=zeros(3,3,n_k);
P(1:3,1:3,1)=P0;
u = [0;0;0]; % initialize with no control input
for k=1:(n_k-1)
    %system dynamics
    w = randn(3,1)*w_stdev; %process noise
    xtrue(:,k+1) = F*xtrue(:,k) + B*u + w;
    %sensor measurement
    v = [randn()*sqrt(R_noise(1,1)); randn()*sqrt(R_noise(2,2))]; %sensor noise 
    z = xtrue(1:2,k) + v;
    % estimate state using Kalman filter
    [xhat_kp1, P_kp1, rej] = HIRO_KF(xhat(:,k), u, P(1:3,1:3,k), z, F, B, G, H, Q, R_filter, Lam0);
    xhat(:,k+1) = xhat_kp1;
    P(1:3,1:3,k+1) = P_kp1;
    % Control Law
    error = x_desired - xhat(:,k+1);
    u = 1*error;
end

% Plot Simulation Results
figure(3); tiledlayout(3, 1)
for i = 1:3
    nexttile; plot(t,xtrue(i,:), t,xhat(i,:), [t(1) t(end)],[x_desired(i) x_desired(i)],'--')
    ylabel(sprintf('%s position (mm)', names(i))); xlabel('time step')
    legend('True Position', 'Estimate', 'setpoint')
    ylim([0 300])
end

