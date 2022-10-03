import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt

rd.seed(0x5eed)

def generate_points_hypercube(n_points, n_dim, radius):
    points = []
    for i in range(n_points):
        point = []
        for j in range(n_dim):
            point.append(rd.uniform(-radius, radius))
        points.append(point)
    return points

def calculate_points_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return distance ** 0.5

def calculate_point_sphere(points, radius):
    num_points_inside_sphere = 0
    for point in points:
        if calculate_points_distance(point, [0] * len(point)) <= radius:
            num_points_inside_sphere += 1

    return num_points_inside_sphere

def generate_plots_for_different_dimensions(n_points, n_dim, radius):
    for i in range(1, n_dim + 1):
        points = generate_points_hypercube(n_points, i, radius)
        num_points_inside_sphere = calculate_point_sphere(points, radius)
        plt.scatter(i, num_points_inside_sphere / n_points, color='red')
    plt.title("Portion of points inside sphere vs. dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Portion of points inside sphere")
    plt.show()

def generate_plots_for_different_radius(n_points, n_dim, radius):
    for i in range(1, radius + 1):
        points = generate_points_hypercube(n_points, n_dim, i)
        num_points_inside_sphere = calculate_point_sphere(points, i)
        plt.scatter(i, num_points_inside_sphere / n_points, color='red')
    plt.title(f"Portion of points inside sphere vs. radius (dimension = {n_dim})")
    plt.xlabel("Radius")
    plt.ylabel("Portion of points inside sphere")
    plt.show()

n_points = 1000
n_dim = 100
radius = 1
generate_plots_for_different_dimensions(n_points, n_dim, radius)
generate_plots_for_different_radius(n_points, n_dim=2, radius=100)
generate_plots_for_different_radius(n_points, n_dim=8, radius=100)