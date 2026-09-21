import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sim.common.quaternion import quat_to_rotmat


def get_box_vertices(half_extents):
    hx, hy, hz = half_extents
    vertices = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ])
    return vertices

def rotate_vertices(vertices, R):
    return vertices @ R.T

def draw_box(ax, vertices, color='blue'):
    edges = [
        [0,1],[1,2],[2,3],[3,0],  
        [4,5],[5,6],[6,7],[7,4],  
        [0,4],[1,5],[2,6],[3,7], ] 
    for edge in edges:
        points = vertices[edge]
        ax.plot(points[:,0], points[:,1], points[:,2], color=color)

def draw_axes(ax, R, length=1.5):
    origin = np.array([0,0,0])
    colors = ['red', 'green', 'blue']

    for i in range(3):
        direction = R[:, i] * length
        ax.quiver(*origin, *direction, color=colors[i])


def animate_plant(plant, torque, dt, steps):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    verts = get_box_vertices([1.0, 0.5, 0.3])
    for i in range(steps):
        plant.step(torque, dt)

        ax.clear()
        q = plant.state[0:4]
        R = quat_to_rotmat(q)
        rotated = rotate_vertices(verts, R)

        draw_box(ax, rotated)
        draw_axes(ax, R)

        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([-2,2])

        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    from sim.plant.rigid_body import RigidBodyPlant

    I = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0]
    ])
    initial_state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plant = RigidBodyPlant(I, initial_state)

    torque = np.array([0.0, 0.0, 0.5])
    animate_plant(plant, torque, dt=0.02, steps=300)
