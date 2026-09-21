import numpy as np
def rk4_step(f, x, dt, *args):
    k1 = f(x, *args)
    k2 = f(x + dt/2 * k1, *args)
    k3 = f(x + dt/2 * k2, *args)
    k4 = f(x + dt * k3, *args)

    x_next = x + (dt/6) * (k1+ 2*k2 + 2*k3 + k4)
    return x_next


if __name__ == "__main__":
    def f_zero(x):
        return 0.0


    x0 = 5.0
    result = rk4_step(f_zero, x0, dt=0.1)
    print(result)

    def f_exp(x):
        return x


    x = 1.0 
    dt = 0.01
    steps = 100

    for i in range(steps):
        x = rk4_step(f_exp, x, dt)

    analytic = 1.0 * np.exp(dt * steps)
    print(x)
    print(analytic)
