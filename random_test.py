import random
import math
def approximate_pi():
    total_points = 0
    within_circle = 0
    for i in range (10000):
        x = random.random()
        y = random.random()
        total_points += 1
        distance = math.sqrt(x**2+y**2)
        if distance < 1:
            within_circle += 1
        if total_points % 1000 == 0:
            pi_estimate = 4 * within_circle / total_points
            yield pi_estimate

pi_estimates = list(es for es in approximate_pi())
errors = list(estimate-math.pi for estimate in approximate_pi())
print(pi_estimates)
print(errors)