%% System definition



% Mass and inertia

m = 5.0;

I = 0.04;



% State-space matrix A (7 states: x, y, theta, dx, dy, dtheta, dummy)

A = [0 0 0 1 0 0;

     0 0 0 0 1 0;

     0 0 0 0 0 1;

     0 0 0 0 0 0;

     0 0 0 0 0 0;

     0 0 0 0 0 0;];



% Input matrix B (3 generalized inputs: Fx, Fy, Torque)

B = [0     0     0;

     0     0     0;

     0     0     0;

     1/m   0     0;

     0   1/m     0;

     0     0   1/I];



% Mapping matrix W (maps tensions to generalized forces/torques)

W = [-0.9015  -0.9993   0.9015   0.9993   0.5293  -0.5279    0.0;

     -0.4327   0.0381  -0.4327   0.0381   0.8484   0.8493    1.0;

      55.1732 -4.8536 -55.1732   4.8536 -68.8105  68.6314    0.0];



% Effective input matrix: maps tensions to state dynamics

Beff = B * W;  % 7x7



% Weight matrices for LQR cost

Q = diag([100, 100, 100, 10, 10, 10]);    % Penalize x, y, theta and their velocities

R = eye(7) * 0.1;                         % Penalize tension magnitudes



% Compute infinite-horizon LQR gain using MATLAB's dlqr

[K, P, ~] = dlqr(A, Beff, Q, R);



%% Forward simulation



N = 50;                     % Time horizon

x0 = [0.1; -0.1; 0.3; 0; 0; 0];  % Initial state

x = zeros(6, N+1);

u = zeros(7, N);

x(:,1) = x0;



for k = 1:N

    u(:,k) = -K * x(:,k);                        % Apply LQR control

    x(:,k+1) = A * x(:,k) + Beff * u(:,k);       % System update

end



% Compute cost

J = x(:,1)' * P * x(:,1);

fprintf('Total optimal cost J(u): %.4f\n', J);



%% Plot state trajectories



% Plot state trajectories

figure;

plot(0:N, x(1,:), '-o', 'LineWidth', 1.5);

grid on;

xlabel('Time Step'); ylabel('x [m]');

title('State Trajectory - x');



figure;

plot(0:N, x(2,:), '-o', 'LineWidth', 1.5);

grid on;

xlabel('Time Step'); ylabel('y [m]');

title('State Trajectory - y');



figure;

plot(0:N, x(3,:), '-o', 'LineWidth', 1.5);

grid on;

xlabel('Time Step'); ylabel('Orientation [rad]');

title('State Trajectory - Orientation');