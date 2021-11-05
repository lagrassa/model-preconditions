import numpy as np
from numba import jit

# Numba gives a false positive warning when multiplying matrices of sizes 1x1
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@jit(nopython=True, cache=True)
def ltv_lqr(F, f, C, c, n, m, T):
    # Backward recursion
    K = np.zeros((T + 1, m, n))
    k = np.zeros((T + 1, m, 1))

    Vtp1 = np.zeros((n, n))
    vtp1 = np.zeros((n, 1))

    Ft = np.zeros((n, n + m))
    ft = np.zeros((n, 1))
  
    for t in range(T, -1, -1):
        Qt = C[t] + Ft.T @ Vtp1 @ Ft
        qt = c[t] + Ft.T @ Vtp1 @ ft + Ft.T @ vtp1

        K[t] = -np.linalg.inv(Qt[n:,n:]) @ Qt[n:,:n]
        k[t] = -np.linalg.inv(Qt[n:,n:]) @ qt[n:,:]
        
        if t > 0:
            Vtp1 = Qt[:n,:n] + Qt[:n, n:] @ K[t] + K[t].T @ Qt[n:, :n] + K[t].T @ Qt[n:,n:] @ K[t]
            vtp1 = qt[:n] + Qt[:n, n:] @ k[t] + K[t].T @ qt[n:, :] + K[t].T @ Qt[n:,n:] @ k[t]
            
            Ft[:] = F[t-1]
            ft[:] = f[t-1]

    return K, k


@jit(nopython=True, cache=True)
def populate_cost_waypoints(waypoints, dt, T, Q, QR, RQ, R, n, m):
    waypoints = np.expand_dims(waypoints, 2)
    C_all = np.zeros((T + 1, n + m, n + m))
    C_all[:] = np.append((np.append(np.zeros((n, n)), QR, axis=1)), (np.append(RQ, R, axis=1)), axis=0)

    # cx = -Q @ xf
    cu_all = np.zeros((T + 1, m, 1))
    # ct = np.append(cx, cu, axis=0)
    
    c_all = np.zeros((T + 1, n + m, 1))
    const_all = np.zeros(T + 1)

    sig = 1.0
    n_wp = waypoints.shape[1]
    t_wp = np.arange(1, n_wp + 1) * T / (n_wp + 1)

    const = 1 / (np.sqrt(2 * np.pi) * sig)
    
    W_ = Q.copy()
    W_[int(n/2):, int(n/2):] = 0
    Wp = np.append((np.append(W_, QR, axis=1)), (np.append(RQ, 0 * np.eye(m), axis=1)), axis=0)
    Wp_all = np.zeros((T + 1, n + m, n + m))
    Wp_all[:] = Wp

    ts = np.arange(T + 1)
    for i in range(n_wp):
        weights = const * np.exp(-((ts - t_wp[i]) * dt) ** 2 / (2 * sig))
        C_all += weights.reshape(-1, 1, 1) * Wp_all

        Wp_waypoints_prod = Wp[:n, :n] @ waypoints[:, i]

        c_all[:, :n] += np.expand_dims((-weights * Wp_waypoints_prod).T, 2)
        c_all[:, n:] += cu_all

        const_all += weights * (waypoints[:, i, 0] @ Wp_waypoints_prod)

    return C_all, c_all, const_all
