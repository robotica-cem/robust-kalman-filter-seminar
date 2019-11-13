%% Simulate tracking problem. Estimate state with spiky noise. Compare kf and rkf 
% Model
%   x(k+1) = [1  h; 0 0]x(k) + Vv(k)
%     y(k) = [1 0]x(k) + e(k) + w(k)
% True system is moving in a circular path

rng(421); % Set random generator seed to get reproducible simulation results

% Flags to set what to simulate
OUTLIERS = 1;  % Set to 1 for Bernoulli-Gaussian higher than 1 means that sample is outlier
ROBUSTKF = 1; % Set to 1 to use rkf
lambda = 6; % Parameter to control the width of the quadratic region in the Huber penalty
p_outlier = 0.1; % The probability of an outlier

% The model - continuous-time constant velocity model for non-manouvring
% target, unit sampling time
h = 1; % The sampling time
F = [eye(2), h*eye(2); zeros(2,2), eye(2)];
V = [eye(2)*h^2/2; eye(2)*h];
H = [eye(2), zeros(2,2)]; % Observing position
D = 0;

%Data
N = 100;
n = 4; % The number of states
m = 2; % The number of measurements
td = 1:N;
radius = 2;
omega = 2*pi/100; % One rotation in 100 sampling periods
xTrue = radius*[cos(td*omega); sin(td*omega); -omega*sin(td*omega); omega*cos(td*omega)];

sigmaV = 0.01; % Process noise
sigmaE = 0.2; % Measurement noise
Q = sigmaV^2*eye(2);
R = sigmaE^2*eye(2);
v = randn(1,N)*sigmaV; % Process noise
e = randn(m,N)*sigmaE; % Measurement noise

if OUTLIERS==1
    % Add bernoulli-gaussian noise
    n_a = sigmaE*10;     % Amplitude of outlier
    bg_noise = n_a*randn(size(e));
    bg_noise(find(rand(size(e)) > p_outlier)) = 0;
    e = e + bg_noise;
    zEstim = zeros(size(e));
    xRKF = zeros(n,N);
elseif max(OUTLIERS) > 1
    n_a = sigmaE*10;     % Amplitude of outlier
    e(:,OUTLIERS) = n_a*randn(2, length(OUTLIERS));
    xRKF = zeros(n,N);
end
% Run kf

% Variables to store results
y = zeros(m,N);
xEst = zeros(n,N);
% Initial values
xNk = zeros(n,1);
xEk = zeros(n,1);
xEkk = zeros(n,1);
xMk = zeros(n,1);
Pk = 100*eye(n);
Pkk = 100*eye(n);
    
for k = td
   % Simulate measurements 
   y(:,k) = H*xTrue(:,k) + e(:,k);
   %y(:,k) = C*xTrue(:,k);
        
   % Update the KF
   if ROBUSTKF
       [xEkpred, Pkpred, xEkk, Pkk] = rkf(F, zeros(4,1), V, H, D, Q, R, xEkk, Pkk, 0, y(:,k), lambda);
       xRKF(:,k) = xEkk;
   end
   [xEkpred, Pkpred, xEk, Pk, K] = kf(F, zeros(4,1), V, H, D, Q, R, xEk, Pk, 0, y(:,k));
   
   xEst(:,k) = xEk;
end

% Plot
figure(1)
clf

if ROBUSTKF
    plot(xTrue(1,:), xTrue(2,:), '-k');
    hold on
    plot(y(1,:), y(2,:), 'ro');
    plot(xEst(1,:), xEst(2,:), 'g');
    plot(xRKF(1,:), xRKF(2,:), 'm');
    xlabel('x')
    ylabel('y')
    legend('True', 'Measured', 'KF', 'Robust KF', 'location', 'southeast');
    title ('Target position')
    axis equal    
    print -dpdf circular_movement_rkf.pdf
else
    plot(xTrue(1,:), xTrue(2,:), '-k');
    hold on
    plot(y(1,:), y(2,:), 'ro');
    plot(xEst(1,:), xEst(2,:), 'g');
    xlabel('x')
    ylabel('y')
    legend('True', 'Measured', 'KF', 'location', 'southeast');
    title ('Target tracking')
    axis equal
    print -dpdf circular_movement_kf.pdf
end
