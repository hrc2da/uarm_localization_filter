%% Setup
clc,clear
close all
rng(100)  %this creates repeatable random numbers

%% System simulation

names = ['x' 'y' 'z']; %state names to be used for plots

% Discrete time system matrices
F = eye(3);
B = eye(3);
H = [1 0 0;
     0 1 0];
D = 0;
system = ss(F, B, H, D, 1);
% define sample time as 1 second for simplicity

% Simulation
x0 = [0; 150; 150]; % default starting position of uArm
t = [0:20]; %time steps
n_k = length(t); % number of time steps
u = randi([-20,20], 3, n_k); % generate random control inputs

[Y,T,X] = lsim(system,u,t,x0);    % without process noise
x_clean = X';
% Generate noise
w_stdev = 12;  %standard deviation of process noise
Q = w_stdev^2;
w = randn(3, n_k)*w_stdev;

%x_noisy = lsim(system,u+w,t,x0)'; % both disturbance and input are additive
x_noisy = x_clean+w; % or disturbances are additive line this?

% Plot Simulation Results
figure(1); tiledlayout(3, 1)
for i = 1:3
    nexttile; plot(t,x_clean(i,:), t,x_noisy(i,:))
    ylabel(sprintf('%s position (mm)', names(i))); xlabel('time step')
    legend('Ideal Position', 'True Position')
end

%% Sensor Simulation

R = [10^2 0; % guess for variance on noise for camera measurement
     0 10^2];
v = [randn(1, n_k)*sqrt(R(1,1));
     randn(1, n_k)*sqrt(R(2,2))];
z = x_noisy(1:2,:)+v;

% Plot Simulation Results
figure(2); tiledlayout(2, 1)
for i = 1:2
    nexttile; plot(t,x_noisy(i,:), t,z(i,:))
    ylabel(sprintf('%s position (mm)', names(i))); xlabel('time step')
    legend('True Position', 'Sensor Reading')
end

%% Filter

G = eye(3); % process noise is simply additive
P0 = [50^2 0 0;
      0 50^2 0;
      0 0 100^2];

Pgate = 0.90;
Lam0 = chi2inv(Pgate, 2);   %statistic threshold
rej=0;  % initialize number of rejected values

%initialize estimate values
xhat=zeros(3,n_k);
xhat(:,1)=x0;
P=zeros(3,3,n_k);
P(1:3,1:3,1)=P0;

for k=1:(n_k-1)
    [xhat_kp1, P_kp1, rej] = HIRO_KF(xhat(:,k), u(:,k), P(1:3,1:3,k), z(:,k+1), F, B, G, H, Q, R, Lam0);
    xhat(:,k+1) = xhat_kp1;
    P(1:3,1:3,k+1) = P_kp1;
end

% Plot Simulation Results
figure(3); tiledlayout(3, 1)
for i = 1:3
    nexttile; plot(t,x_noisy(i,:), t,xhat(i,:))
    ylabel(sprintf('%s position (mm)', names(i))); xlabel('time step')
    legend('True Position', 'Estimate')
end


% Error
error = xhat - x_noisy;

% σ values
sigma = [squeeze(sqrt(P(1,1,:)))';
         squeeze(sqrt(P(2,2,:)))';
         squeeze(sqrt(P(3,3,:)))'];

figure(4); tiledlayout(3, 1)

for i = 1:3
    nexttile; plot(t,error(i,:), t,2*sigma(i,:),'g', t,-2*sigma(i,:),'g')
    ylabel(sprintf('%s position (mm)', names(i))); xlabel('time step')
    legend('Error', '2σ bounds')
end