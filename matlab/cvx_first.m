%% cvx-test
% Kjartan Halvorsen
% 2015-10-29

% Generate a signal according to a simple linear model y=ax+b. We will
% estimate a and b from noisy measurements.
a0 = 2;
b0 = 1;
N = 200;
x = linspace(0,10,N)';
unos = ones(size(x));
y0 = a0*x + b0*unos; 

% Add some noise
% First white gaussian noise
sgma = 4;
yg = y0 + sgma*randn(size(y0));

% Then also with bernoulli-gaussian noise
p_outlier = 0.1; % Probability of outlier
n_a = sgma*4;     % Amplitude of outlier
bg_noise = abs(n_a*randn(size(y0)));
bg_noise(find(rand(size(y0)) > p_outlier)) = 0;

ybg = yg + bg_noise;

figure(1)
clf
plot(x, y0, 'b')
hold on
plot(x, yg, 'ro')
plot(x, ybg, 'm.')

%% Least squares estimate of parameters a and b

% First Least-squares, also known as l2-minimization:
%    minimize_{a,b}  f(a,b) = 1/2 ||y - (a*x+b)||_2
% which can be written
%    minimize_{e,a,b} 1/2 ||e||_2
%    such that   e = y - (a*x+b*1),
%   1 is a column vector of ones.

% The solution is found by taking the derivative of the criterion function
% and setting equal to zero:
% Let z = [a;b] be the variables in the problem
%    d/dx 1/2 ||e||_2 = d/dx 1/2 e'*e = e' * d/x e = e' * [-x, -1] = (y-(a*x-b*1))' * [-x, -1] = 0
%    => 2 equations
%    (1) -y'*x + a*x'*x + b*1'*x = 0 (scalar)
%    (2) -y'*1 + a*x'*1 + b*1'*1 = 0 (scalar)
%    Write (1) and (2) as linear equations in a and b:
%    [x'*x  x'*1  ] * [a;b] = [x'*y']
%    [1'*x  1'*1  ]           [1'*y']
%    [x 1]' * [x 1] * [a;b] = [x 1]'*y
%     A'*A * x = A'*y
A = [x unos];
zhat2m = A\ybg  % Least squares solution using matlab's builtin solver
zhat2n = inv(A'*A)*A'*ybg % Solve directly (should be avoided because of numerical issues)
a2 = zhat2m(1);
b2 = zhat2m(2);
sprintf('True parameters: a=%f, b=%f', a0, b0)
sprintf('Least squares estimates: a=%f, b=%f', a2, b2)


% The same problem solved with cvx
cvx_begin
    variable z2(2)
    minimize( norm(A*z2-ybg) )
cvx_end
sprintf('Least squares estimates (cvx): a=%f, b=%f', z2(1), z2(2))

% Plot of residuals
e2 = ybg - (a2*x + b2*unos);
figure(2)
clf
hist(e2, 50)

% Plot regression line
figure(1)
l2r = plot(x, a2*x + b2*unos, 'g', 'linewidth', 2)

%% L1-minimization
% Instead of minimizing the sum of square of the residuals or the l2-norm (which is what
% least squares does), we can minimize the absolute value of the residuals
% instead. This is the same as minimizing the l1-norm of the residuals. 
% This cannot be done in closed form. However, the problem is a convex optimization problem and 
% can be solved fast and (extremely) simply with cvx:

cvx_begin
    variable z1(2)
    minimize( norm(A*z1-ybg, 1) ) % Note the 1!
cvx_end
a1 = z1(1);
b1 = z1(2);
sprintf('l1-minimizaion estimates (cvx): a=%f, b=%f', a1, b1)

% Plot of residuals
e1 = ybg - (a1*x + b1*unos);
figure(3)
hist(e1, 50)

% Plot regression line
figure(1)
l1r = plot(x, a1*x + b1*unos, 'c', 'linewidth', 2)

%% L1 - regularization
% Let's go back to the model. We had
%   y = ax + b + v + w,
% where v is a white Gaussian noise and w is a sparse vector (meaning most of its elements are zer)
% with outliers (Bernoulli-Gaussian noise in this case).
% Write the model as
%    y-(ax+b)-w = v
% If we know the outlier vector w, we can subtract this and get a residual
% vector which is Gaussian. Such a vector is best minimized using least
% squares. 
% This leads us to the following optimization problem
%   minimize 0.5 * (y - (ax + b) - w)'*(y - (ax + b) - w)   +   gamma*||w||_1
% with optimization variables z=[a;b;w].
% Including the l1-norm of w in the criterion is a powerful trick. The
% l1-norm will make sure the w vector is sparse, depending on the tuning
% variable gamma. With gamma=0, then w will not be sparse and it will contain the 
% residuals for the least squares solution to the original problem.
% Write the problem as 
%   minimize  0.5*(AA*z - y)'*(AA*z - y) + gamma* ||C*z||_1, 
% where
%   AA = [A I_n] (N x (2+N)), and
%    C = [0 I_n] ( N x (2+N))
%
% The criterion is actually is equivalent to 
%   minimize  norm(AA*z - y)) + gamma2* ||C*z||_1, 
% but with a different value of gamma2.
%
% if we know that the white gaussian noise has a certain variance (or
% covariance if it's multicariable), then the quadratic term in our
% minimzation problem can be weighted by the inverse of this variance. This
% gives
%   minimize  0.5*(AA*z - y)'*W*(AA*z - y) + gamma * ||C*z||_1, 
% or, equivalently
%   minimize  norm(sqrt(W)*(AA*z - y)) + gamma2 * ||C*z||_1, 
%
% where W is the inverse of the covariance matrix for the noise vector v. Since this noise
% is white (uncorrelated) and time invariant with variance sgma^2, 
% W is a diagonal matrix with 1/sgma^2 on the diagonal. The criterion can
% then be written 
%   minimize  norm((1/sgma)*(AA*z - y)) + gamma2* ||C*z||_1, 
%  

AA = [A eye(N)];
C = [zeros(N,2) eye(N)]; 
gamma = 0.5;
%gamma2 = 0.04;
cvx_begin
    variable zl1(2+N)
    minimize( (0.5/sgma^2)*(AA*zl1 -ybg)'*(AA*zl1 -ybg) + gamma*norm(C*zl1, 1))
    %minimize( norm( (1/sgma)*(AA*zl1 -ybg) ) + gamma2*norm(C*zl1, 1) )
cvx_end
al1 = zl1(1);
bl1 = zl1(2);
w = zl1(3:N+2);
tol = 1e-6;
outliers = find(abs(w) > tol);

sprintf('l1-regularized least squares estimates (cvx): a=%f, b=%f', al1, bl1)
sprintf('Number of outliers: %d', length(outliers))
max(w)

% Plot histogram of residuals
el1 = ybg - (al1*x + bl1*unos + w);
figure(4)
hist(el1, 50)

% Plot regression line and identify outliers
figure(1)
l1rr = plot(x, al1*x + bl1*unos, 'y', 'linewidth', 2)
for i=find(abs(w)>tol)
  plot(x(i), al1*x(i) + bl1 + w(i), 'yo')
end

%% L1-regularization as a Quadratic Programming problem
%
% By inroducing w=wp-wm, withe wp and wm non-negative we can write the l1-regularization problem
%   minimize  0.5(Ax + w -y)'W(Ax + w - y)  + gamma ||w||_1
% as 
%   minimize 0.5*([A I -I]*[x;wp;wm] - y)'*W*([A I -I]*[x;wp;wm] - y) 
%               + gamma*(sum(wp) + sum(wm) )
%   subject to
%              wp >= 0
%              wm >= 0
% This problem has the form
%    minimize z'*Q*z + c'z
%    subject to
%             Fz >= 0
% with z = [x;wp;wm]:
% Write the criterion function as 
%    0.5(B*z-y)̈́'*W*(B*z-y) + gamma*[0 I I]*z  
%           = 0.5*z'*B'*W*B*z - y'*W*B*z + 0.5*y'*W*y + gamma*[0  I I ] z
% Since the term 0.5*y'*W*y does not depend on z, we can eliminate it from the
% criterion function and obtain
%    minimize 0.5*z'*B'*W*B*z + (-y'*W*B + gamma*[0  I I ])* z
%                   = z'* Q * z + c'*z
%    subject to
%             [0 I 0;
%              0 0 I] * z >= 0
gamma = 0.05;
B = [A eye(N) -eye(N)];
Q = (0.5/sgma^2)*B'*B;
cc = -ybg'*B/sgma^2 + gamma*[zeros(1,2) ones(1,2*N)];
F = [zeros(N,2) eye(N) zeros(N,N); zeros(N,2) zeros(N,N) eye(N)];
cvx_begin
   variable zqp(2+2*N,1)
   minimize( cc*zqp + zqp'*Q*zqp )
   subject to
        F*zqp >= 0;
        
cvx_end

aqp = zqp(1);
bqp = zqp(2);
wp = zqp(3:N+2);
wm = zqp(N+3:2*N+2);
w = wp - wm;
tol = 1e-6;
outliers = find(abs(w) > tol);

sprintf('l1-regularized least squares as QP (cvx): a=%f, b=%f', aqp, bqp)
sprintf('Number of outliers: %d', length(outliers))
max(w)

% Plot histogram of residuals
eqp = ybg - (aqp*x + bqp*unos + w);
figure(5)
hist(eqp, 50)

% Plot regression line and identify outliers
figure(1)
l1pqr = plot(x, aqp*x + bqp*unos, 'b', 'linewidth', 2);
for i=find(abs(w)>tol)
  plot(x(i), aqp*x(i) + bqp + w(i), 'bo')
end
legend([l2r, l1r, l1rr, l1pqr], 'Least-squares', 'l1-minimization', 'l1-regularization', 'l1-reg as QP')
