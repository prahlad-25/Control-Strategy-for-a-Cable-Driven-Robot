import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

horizon = [0, 5]

X0 = np.array([0, 0, 1, 0.02, 0.02, 0.02, 0, 0, 0, 0, 0, 0])
Xd = np.array([0.1,-0.1, 1.3, 0, 0, 0])
Xddot_d = np.zeros(6)

Kp = np.diag([20, 20, 20, 10, 10, 10])
Kd = np.diag([10, 10, 10, 3, 3, 3])

print(f"X0={X0}")
print(f"Xd={Xd}")

def compute_jacobian(X):

    pos = X[:3]
    r = X[3]        # roll
    p = X[4]        # pitch
    y = X[5]        # yaw

    R = Rotation.from_euler('zyz', [y, p, r]).as_matrix()

    A = np.array([[-127.5, 127.5, 0], [0, 0, 130], [0, 0, 0]])                                                                                     #   anchor points on end-effector
    P = np.array([[-390, -390, 390, 390, 138, -138, 0], [-126, 10, -126, 10, 352, 352, 352], [-196, 394, -196, 394, -196, -196, 394]])      #   anchor points on frame

    attach_map = np.array([0, 0, 1, 1, 2, 2, 2])       # indices

    # initialize jacobian
    J = np.zeros((7, 6))

    cable_dirs = np.zeros((3, 7))
    cable_lengths = np.zeros((1, 7))

    for i in range(7):
        A_local = A[:, attach_map[i]]
        A_world = pos + R @ A[:, attach_map[i]] # anchor points in world coordinates

        l = P[:,i] - A_world            # cable vector
        l_dir = l / np.linalg.norm(l)   # cable dir

        J[i, :3] = -l_dir
        J[i, 3:7] = -np.cross(l_dir, R @ A_local)

        cable_dirs[:, i] = l_dir
        cable_lengths[:, i] = np.linalg.norm(l)

    return J

def dynamics_ode(t, X, Xd, Xddot_d, Kp, Kd):

    X_pos = X[:6]
    X_vel = X[6:]

    e_pos = X_pos[:3] - Xd[:3]          # position error
    e_orient = X_pos[3:] - Xd[3:]       # orientation error

    ed_pos = X_vel[:3]                  # velocity error
    ed_orient = X_vel[3:]               # angular velocity error

    v_pos = Xddot_d[:3] - Kd[0, 0]*ed_pos -Kp[0, 0]*e_pos           # position control
    v_orient = Xddot_d[3:] - Kd[1, 1]*ed_orient -Kp[1, 1]*e_orient  # orientation control

    v = np.concatenate((v_pos, v_orient))

    m = 5
    I = [0.4, 0.4, 0.4]
    M= np.diag([m, m, m] + I)

    C = np.zeros((6, 6))
    G = np.array([0, 0, -m * 9.81, 0, 0, 0])

    tau = np.dot(M, v) + np.dot(C, X_vel) + G

    J = compute_jacobian(X_pos)

    res = lsq_linear(J.T, -tau, bounds=(0, np.inf))
    u = res.x

    dXdt = np.zeros(12)
    dXdt[:6] = X_vel
    dXdt[6:] = np.linalg.solve(M, (-np.dot(C, X_vel) - G - np.dot(J.T, u)))

    return dXdt, u

solution = integrate.solve_ivp(lambda t, X, Xd, Xddot_d, Kp, Kd: dynamics_ode(t, X, Xd, Xddot_d, Kp, Kd)[0], horizon, X0, args=(Xd, Xddot_d, Kp, Kd), dense_output=True)

t = solution.t
Xsol = solution.y.T
X_traj = Xsol[:, :6]
Xdot_traj = Xsol[:, 6:]
#print(f"Xtraj={X_traj}")

# Initialize an array to store tensions for each cable
U = np.zeros((len(t), 7))

for i in range(len(t)):
    dXdt, u_i = dynamics_ode(t[i], Xsol[i, :], Xd, Xddot_d, Kp, Kd)  # Get both state and tensions
    U[i, :] = u_i


plt.figure(figsize=(10, 8))

# Position Plots (x, y, z)
plt.subplot(3, 2, 1)
plt.plot(t, X_traj[:, 0], 'r', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Position x [m]')
plt.title('End-Effector Position (x)')

plt.subplot(3, 2, 2)
plt.plot(t, X_traj[:, 1], 'g', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Position y [m]')
plt.title('End-Effector Position (y)')

plt.subplot(3, 2, 3)
plt.plot(t, X_traj[:, 2], 'b', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Position z [m]')
plt.title('End-Effector Position (z)')

# Orientation Plots (roll, pitch, yaw)
plt.subplot(3, 2, 4)
plt.plot(t, X_traj[:, 3], 'r', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Roll [rad]')
plt.title('End-Effector Orientation (Roll)')

plt.subplot(3, 2, 5)
plt.plot(t, X_traj[:, 4], 'g', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Pitch [rad]')
plt.title('End-Effector Orientation (Pitch)')

plt.subplot(3, 2, 6)
plt.plot(t, X_traj[:, 5], 'b', linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Yaw [rad]')
plt.title('End-Effector Orientation (Yaw)')

plt.tight_layout()
plt.show()

# Loop through each cable and plot its tension profile
for i in range(7):
    plt.figure(figsize=(10, 6))  # Create a new figure for each tension plot
    plt.plot(t, U[:, i], label=f'Tension T{i + 1}', linewidth=1.5)
    plt.xlabel('Time [s]')
    plt.ylabel(f'Tension T{i + 1} [N]')
    plt.title(f'Cable {i + 1} Tension')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
