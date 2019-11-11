%% Simulate tracking problem. Estimate state with spiky noise. Compare kf and rkf 
% Model
%   x(k+1) = [1  h; 0 0]x(k) + Fv(k)
%     y(k) = [1 0]x(k) + e(k) + w(k)
% True system is moving in a circular path

% Flags to set what to simulate
OUTLIERS = 10;  % Set to 1 for Bernoulli-Gaussian higher than 1 means that sample is outlier
ROBUSTKF = 1; % Set to 1 to use rkf
lambda = 20; % Parameter to control sparsity of outlier vector in rkf
p_outlier = 0.1; % The probability of an outlier

% The model - continuous-time constant velocity model for non-manouvring
% target, unit sampling time
h = 1; % The sampling time
Hm = [eye(2), h*eye(2); zeros(2,2), eye(2)];
Fm = [eye(2)*h^2/2; eye(2)*h];
C = [eye(2), zeros(2,2)]; % Observing position
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
sigmaE = 0.1; % Measurement noise
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
elseif OUTLIERS > 1
    e(:,OUTLIERS) = 2*[1;1];
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
   % Simualate measurements 
   y(:,k) = C*xTrue(:,k) + e(:,k);
   %y(:,k) = C*xTrue(:,k);
        
   % Update the KF
   if ROBUSTKF
       [xEkpred, Pkpred, xEkk, Pkk, zk] = rkf2(Hm, zeros(4,1), Fm, C, D, Q, R, xEkk, Pkk, 0, y(:,k), lambda);
       zEstim(:,k) = zk;
       xRKF(:,k) = xEkk;
   end
   [xEkpred, Pkpred, xEk, Pk, K] = kf(Hm, zeros(4,1), Fm, C, D, Q, R, xEk, Pk, 0, y(:,k));
   
   xEst(:,k) = xEk;
end

% Plot
figure(1)
clf

if ROBUSTKF
    % Find where the outliers are
    outlind = find (abs(zEstim(1,:)) > 1e-2);
    display( sprintf('Number of outliers: %d', length(outlind)) )
    plot(xTrue(1,:), xTrue(2,:), '-k');
    hold on
    plot(y(1,:), y(2,:), 'r.');
    plot(xEst(1,:), xEst(2,:), 'g');
    plot(xRKF(1,:), xRKF(2,:), 'm');
    plot(xRKF(1,outlind) + zEstim(1,outlind), xRKF(2,outlind) + zEstim(2,outlind), 'mo')
    xlabel('x')
    ylabel('y')
    legend('True', 'Measured', 'KF', 'Robust KF');
    title ('Target position')
    axis equal    
else
    plot(xTrue(1,:), xTrue(2,:), '-k');
    hold on
    plot(y(1,:), y(2,:), 'r');
    plot(xEst(1,:), xEst(2,:), 'g');
    xlabel('x')
    ylabel('y')
    legend('True', 'Measured', 'KF');
    title ('Target tracking')
    axis equal
end
