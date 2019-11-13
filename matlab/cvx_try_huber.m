%% cvx-test
% Kjartan Halvorsen
% 2015-10-29

close all

rng(124)
% Generate a signal according to a simple linear model y=ax+b. We will
% estimate a and b from noisy measurements.
a0 = 2;
b0 = 1;
N = 40;
x = linspace(0,10,N)';
unos = ones(size(x));
y0 = a0*x + b0*unos; 


% Add some noise
% First white gaussian noise
sgma = 4;
yg = y0 + sgma*randn(size(y0));

% Then also with one-sided bernoulli-gaussian noise
p_outlier = 0.1; % Probability of outlier
n_a = sgma*20;     % Amplitude of outlier
bg_noise = abs( n_a*randn(size(y0)));
bg_noise(find(rand(size(y0)) > p_outlier)) = 0;

ybg = yg + bg_noise;

figure(1)
clf
l0 = plot(x, a0*x + b0, 'k')
hold on
%plot(x, yg, 'b.')
plot(x, ybg, 'ro')
xlabel('x')
ylabel('y')
print -dpdf least_squares_example.pdf

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
cvx_begin quiet
    variable z2(2)
    minimize( norm(A*z2-ybg) )
cvx_end
sprintf('Least squares estimates (cvx): a=%f, b=%f', z2(1), z2(2))

% Plot of residuals
e2 = ybg - (a2*x + b2*unos);
figure(2)
clf
h2 = histogram(e2, 21)
title('Residuals, least-squares')

% Plot regression line
figure(1)
l0 = plot(x, a0*x + b0, 'k')
hold on
l2r = plot(x, a2*x + b2*unos, 'g', 'linewidth', 2)
legend([l0, l2r], sprintf('True, a=%.3f, b=%.3f', [a0,b0]),...
    sprintf('Least squares, a=%.3f, b=%.3f', [a2, b2]))
ylabel('y')
xlabel('x')

print -dpdf least_squares_regression.pdf
binMids = 0.5*(h2.BinEdges(1:end-1)+h2.BinEdges(2:end))
dlmwrite('ls_residuals.dat', cat(1, binMids, h2.Values/sum(h2.Values))', ',')

%% Huber-minimization
% Instead of minimizing the sum of square of the residuals or the l2-norm (which is what
% least squares does), we now minimizethe Huber penalty function.
% This penalizes large residuals with their absolulte values,
% and small residuals with the square.
cvx_begin quiet
    variable z1(2)
    minimize(sum(huber(A*z1-ybg, sgma )))
cvx_end

a1 = z1(1);
b1 = z1(2);
sprintf('Huber minimizaion estimates (cvx): a=%f, b=%f', a1, b1)

% Plot of residuals
e1 = ybg - (a1*x + b1*unos);
figure(3)
hh = histogram(e1, 20)
title('Residuals, Huber regression')

% Plot regression line
figure(11)
clf
hold on
%plot(x, yg, 'b.')
plot(x, ybg, 'ro')

l0 = plot(x, a0*x + b0, 'k');
l2r = plot(x, a2*x + b2*unos, 'g', 'linewidth', 2);
l1r = plot(x, a1*x + b1*unos, 'c', 'linewidth', 2);
legend([l0, l2r,l1r], sprintf('True, a=%.3f, b=%.3f', [a0,b0]),...
    sprintf('Least squares, a=%.3f, b=%.3f', [a2, b2]),...
    sprintf('Huber, a=%.3f, b=%.3f', [a1, b1]))
ylabel('y')
xlabel('x')
print -dpdf robust_least_squares_regression.pdf

binMids = 0.5*(hh.BinEdges(1:end-1)+hh.BinEdges(2:end))
dlmwrite('huber_residuals.dat', cat(1, binMids, hh.Values/sum(hh.Values))', ',')
