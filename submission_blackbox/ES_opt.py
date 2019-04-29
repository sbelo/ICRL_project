import numpy as np

class Result:
    def __init__(self,theta,val,iter,done):
        self.x = theta
        self.fun = val
        self.nfev = iter
        self.done = done


def ES_minimize(target_func, init_step, alpha, sigma, num_eps, theta_init, tol, stop_cond, sig_decay, maxiter, tol_stop=1e-10 ):
    theta_t = theta_init
    theta_min = theta_t
    val = target_func(theta_t)

    if val < tol_stop:
        return Result(theta_init,val,0,True)

    sig = 1
    min = val
    new_val = val
    stop_ind = 0
    for t in range(maxiter):


        if stop_ind >= stop_cond:
            break

        weighted_sum = 0
        for i in range(int(num_eps/2)):
            eps = np.random.normal(size=theta_init.shape)

            feval = target_func(theta_t + (sig*sigma * eps))
            n_feval = target_func(theta_t - (sig*sigma * eps))
            weighted_sum += feval*eps - n_feval*eps

        theta_t += (-1)*init_step*np.exp(-alpha*t)*(1.0/(num_eps*sig*sigma))*weighted_sum
        new_val = target_func(theta_t)

        if new_val < min:
            min = new_val
            theta_min = theta_t

    return Result(theta_min, min, t, False)