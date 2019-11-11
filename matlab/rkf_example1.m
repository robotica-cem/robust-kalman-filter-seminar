%% Simulate RLC circuit. Estimate state with spiky noise. Compare kf and rkf 
% Model
%   x(k+1) = Hx(k) + Gu(k) + Fv(k)
%     y(k) = Cx(k) + Du(k) + e(k)

% Flags to set what to simulate
OUTLIERS = 1;
ROBUSTKF = 1; % Set to 1 to use rkf
lambda = 4.9; % Parameter to control sparsity of outlier vector in rkf
p_outlier = 0.1; % The probability of an outlier

L = 1.0;
C = 1000*10e-6;
R = 30.0;
h = 0.01;
[H, G, Cd, Dd] = rlc(R, L, C, h);
F = G;

% The model used by the KF (change values to simulate imperfect model)
[Hm, Gm, Cdm, Ddm] = rlc(R, L, C, h);
    
%Data
N = 300;
td = 1:N;

sigmaV = 0.5; % Process noise
sigmaE = 0.2; % Measurement noise
Q = sigmaV^2;
R = sigmaE^2;
u = randn(size(td)); % Driving input
v = randn(size(td))*sigmaV; % Process noise
e = randn(size(td))*sigmaE; % Measurement noise

if OUTLIERS
    % Add bernoulli-gaussian noise
    n_a = sigmaE*8;     % Amplitude of outlier
    bg_noise = abs(n_a*randn(size(e)));
    bg_noise(find(rand(size(e)) > p_outlier)) = 0;
    e = e + bg_noise;
    zEstim = zeros(size(e));
    xRKF = zeros(2,N);
end

% Run kf

% Variables to store results
y = zeros(size(td));
xNom = zeros(2,N);
xEst = zeros(2,N);
xMed = zeros(2,N);
% Initial values
xNk = zeros(2,1);
xEk = zeros(2,1);
xEkk = zeros(2,1);
xMk = zeros(2,1);
Pk = eye(2);
Pkk = eye(2);
    
for k = td
   if k<40
      uk = 1;
   else
      if k>80
         uk = 2;
      else
         uk = 0;
      end
   end
   
   u(k) = uk;
   
   % Simualate the nominal state (without process noise) 
   xNk = H*xNk + G*uk;
   xNom(:,k) = xNk;
   
   % ... and the measurement
   xMk = H*xMk + G*uk + F*v(k);
   xMed(:,k) = xMk;
   y(k) = Cd*xMk + e(k);
        
   % Update the KF
   if ROBUSTKF
       [xEkpred, Pkpred, xEkk, Pkk, zk] = rkf(Hm, Gm, Gm, Cd, Dd, Q, R, xEkk, Pkk, uk, y(k), lambda);
       zEstim(k) = zk;
       xRKF(:,k) = xEkk;
   end
   [xEkpred, Pkpred, xEk, Pk] = kf(Hm, Gm, Gm, Cd, Dd, Q, R, xEk, Pk, uk, y(k));
   
   xEst(:,k) = xEk;
end

% Plot
figure(1)
clf

if ROBUSTKF
    % Find where the outliers are
    outlind = find (abs(zEstim) > 1e-2);
    display( sprintf('Number of outliers: %d', length(outlind)) )
    plot(td, xNom(1,:), '-k');
    hold on
    plot(td, y, 'r');
    plot(td, xEst(1,:), 'g');
    plot(td, xRKF(1,:), 'm');
    plot(td(outlind), xRKF(1,outlind) + zEstim(outlind), 'mo')
    xlabel('k')
    ylabel('Respuesta, x(1)')
    legend('Nominal', 'Measured', 'KF', 'Robust KF');
    title ('Kalman filter')
    figure(2)
    clf
    hist(zEstim)
else
    plot(td, xNom(1,:), '-k');
    hold on
    plot(td, y, 'r');
    plot(td, xEst(1,:), 'g');
    xlabel('k')
    ylabel('Respuesta, x(1)')
    legend('Nominal', 'Medido', 'Estimado');
    title ('Kalman filter')
end
