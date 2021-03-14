import numpy as np
import matplotlib.pyplot as plt
from optimization.gradient_descent import gradient_descent
from optimization.newton_method import newton


###########
# STARTING DATA
x0 = -30
y0 = 30
alpha = 0.9
number_of_iterations = 100
epsilon = 1e-8
##########

gradient_coords, gradient_steps, f_values_gradient, gradient_xs, gradient_ys = gradient_descent(x0, y0, alpha, number_of_iterations, epsilon)

newton_coords, newton_steps, f_values_newton, newton_xs, newton_ys = newton(x0, y0, alpha, number_of_iterations, epsilon)

# Prints coordinate of last point
print(gradient_coords[len(gradient_coords) - 1][0], gradient_coords[len(gradient_coords) - 1][1])

print(newton_coords[len(newton_coords) - 1][0], newton_coords[len(newton_coords) - 1][1])


### FIGURES FOR GRADIENT DECENT METHOD
# Initiate space [-5.0, 5.0] x [-5.0, 5.0]
x = np.linspace(-5.0, 5.0)
y = np.linspace(-5.0, 5.0)

xx, yy = np.meshgrid(x, y) # Grid of function with two variables

z = (1 - xx)**2 + 100*((yy - xx**2)**2) # z - Rosenbrock function

levels = [0.0, 50.0, 300.0, 1200.0, 3000.0, 10000.0, 20000.0, 40000.0, 100000.0] # Levels of grid

fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)

# Fills figure with colors depending of value of function
cp = ax[0].contour(xx, yy, z, levels, colors='black', linestyles='dashed', linewidths=1)
ax[0].clabel(cp, inline=1, fontsize=9)
cp = ax[0].contourf(xx, yy, z, levels, alpha=0.8)

# Adds dashed line in point (x, y) = (1, 1) where global minimum of Rosenbrock function occurs
ax[0].axhline(1, color='black', alpha=0.8, dashes=[2, 4],linewidth=1)
ax[0].axvline(1, color='black', alpha=0.8, dashes=[2, 4],linewidth=1)

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

# Makes line from starting point in first figure 
for i in range(len(gradient_coords) - 1):
    ax[0].annotate('', xy=gradient_coords[i], xytext=gradient_coords[i - 1],
                   arrowprops={'arrowstyle': '-', 'color':'w'}, va='center', ha='center')

ax[1].scatter(gradient_xs, gradient_ys, s=10, c='red', edgecolor='black', linewidth=0.5, alpha=0.75)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

ax[2].scatter(gradient_steps, f_values_gradient, s=10, c=f_values_gradient, edgecolor='black', linewidth=0.5, alpha=0.75)

ax[2].set_xlabel('number of steps')
ax[2].set_ylabel('value of loss function')

plt.suptitle('Gradient Descent')

ax[1].grid()
ax[2].grid()
plt.show()


### FIGURES FOR NEWTON METHOD

x = np.linspace(-1.5, 1.5)
y = np.linspace(-1.0, 2.0)

xx, yy = np.meshgrid(x, y)
z = (1 - xx)**2 + 100*((yy - xx**2)**2)

levels = [0.0, 10.0, 50.0, 150.0, 300.0, 600.0, 800.0, 1200.0]

fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False)

cp = ax[0].contour(xx, yy, z, levels, colors='black', linestyles='dashed', linewidths=1)
ax[0].clabel(cp, inline=1, fontsize=9)
cp = ax[0].contourf(xx, yy, z, levels, alpha=0.8)

ax[0].axhline(1, color='black', alpha=0.8, dashes=[2, 4],linewidth=1)
ax[0].axvline(1, color='black', alpha=0.8, dashes=[2, 4],linewidth=1)

ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

for i in range(len(newton_coords) - 1):
    ax[0].annotate('', xy=newton_coords[i], xytext=newton_coords[i - 1],
                   arrowprops={'arrowstyle': '-', 'color':'w'}, va='center', ha='center')

ax[1].scatter(newton_xs, newton_ys, s=20, c='red', edgecolor='black', linewidth=0.5, alpha=0.75)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

ax[2].scatter(newton_steps, f_values_newton, s=20, c=f_values_newton, edgecolor='black', linewidth=0.5, alpha=0.75)

ax[2].set_xlabel('number of steps')
ax[2].set_ylabel('value of loss function')

plt.suptitle('Newton Method')

ax[1].grid()
ax[2].grid()
plt.show()
