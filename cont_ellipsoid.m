T = 10;            % time horizon (seconds)
dt = 0.01;         % time step
steps = round(T/dt);

Wc = zeros(6, 6);  % initialize Gramian

for k = 0:steps
    t = k * dt;
    expAt = expm(A * t);
    Wc = Wc + (expAt * B*W) * (expAt * B*W)' * dt;
end

% Project to xy-plane
Wxy = Wc(1:2, 1:2);

% Generate ellipse
theta = linspace(0, 2*pi, 200);
unit_circle = [cos(theta); sin(theta)];
ellipse = sqrtm(Wxy) * unit_circle;

% Plot
figure;
plot(ellipse(1,:), ellipse(2,:), 'b', 'LineWidth', 2);
hold on;
plot(0, 0, 'rx');
grid on;
axis equal;
xlabel('x'); ylabel('y');
title('Finite-Horizon Controllability Ellipsoid in (x, y)');