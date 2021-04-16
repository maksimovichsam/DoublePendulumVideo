from typing import DefaultDict
from manim import *
import subprocess, sys
import numpy as np
import math

from collections import defaultdict

#############################################
# Generalized N pendulum
#############################################
class Pendulum:
    def __init__(self, N, theta=None):
        self.N = N;
        self.gravity = 9.81
        self.L = np.ones(shape=(N, )) # lengths
        self.m = np.ones(shape=(N, )) # masses
        
        self.theta = np.zeros(shape=(N, )) if theta is None else theta
        self.theta_dot = np.zeros(shape=(N, ))

    def step(self, delta_t):
        equations = np.zeros(shape=(4 * self.N, 4 * self.N + 1))
        theta_dd = 0
        a_xn = theta_dd + self.N
        a_yn = a_xn + self.N
        T_n = a_yn + self.N
        for p in range(0, 4 * self.N, 4):
            n = p // 4
            # First equation x_dd
            equations[p, theta_dd + n] = self.L[n] * math.cos(self.theta[n])
            equations[p, a_xn + n] = -1.0;
            if n > 0:
                equations[p, a_xn + n - 1] = 1
            equations[p, -1] = self.theta_dot[n]**2 * math.sin(self.theta[n])

            # Second equation y_dd
            equations[p + 1, theta_dd + n] = -1.0 * self.L[n] * math.sin(self.theta[n])
            equations[p + 1, a_yn + n] = -1.0
            if n > 0:
                equations[p + 1, a_yn + n - 1] = 1.0
            equations[p + 1, -1] = self.theta_dot[n]**2 * math.cos(self.theta[n])

            # Third equation horizontal tension
            equations[p + 2, a_xn + n] = self.m[n]
            equations[p + 2, T_n + n] = math.sin(self.theta[n])
            if n + 1 < self.N:
                equations[p + 2, T_n + n + 1] = -1.0 * math.sin(self.theta[n + 1])
            equations[p + 2, -1] = 0

            # Fourth equation vertical tension
            equations[p + 3, a_yn + n] = self.m[n]
            equations[p + 3, T_n + n] = math.cos(self.theta[n])
            if n + 1 < self.N:
                equations[p + 3, T_n + n + 1] = -1.0 * math.cos(self.theta[n + 1])
            equations[p + 3, -1] = self.m[n] * self.gravity

        solution = np.linalg.solve(equations[:, 0:-1], equations[:, -1])
        self.theta_dot += delta_t * solution[0:self.N]
        self.theta += delta_t * self.theta_dot


class PhysicalNPendulum(Scene):
    def construct(self):
        p = Pendulum(N=2, theta=[math.pi / 2] * 2)
        scale = 2
        x_c, y_c = 0, 3

        def get_pendulum(dot_radius=DEFAULT_DOT_RADIUS):
            points = [[x_c, y_c, 0]] + [[scale * p.L[n] * math.sin(p.theta[n]), -scale * p.L[n] * math.cos(p.theta[n]), 0] for n in range(p.N)]
            for i in range(1, len(points)):
                for k in range(3):
                    points[i][k] += points[i - 1][k]
            return [ Line(points[i - 1], points[i]) for i in range(1, len(points)) ] + [ Dot(point, radius=dot_radius) for point in points]

        def step(pendulum, dt):
            for obj in pendulum:
                pendulum.remove(obj)
            p.step(dt / 2)
            for obj in get_pendulum(dot_radius=0.06):
                pendulum.add(obj)

        pendulum = VGroup()
        for obj in get_pendulum():
            pendulum.add(obj)
        pendulum.add_updater(step)
        self.add(pendulum)
        self.wait(5)
        pendulum.remove_updater(step)


################################################
# Optimized collection of 2-Pendulum's
################################################
class DoublePendulum():
    """ Creates a collection of 2-Pendulums
        Parameters:
        theta: np.ndarray with shape (N, 2)
        lengths (optional): np.ndarray with shape (N, 2) default is all ones
        masses  (optional): np.ndarray with shape (N, 2) default is all ones
    """
    def __init__(self, theta, lengths=None, masses=None):
        self.t = theta
        self.t_d = np.zeros(theta.shape)
        self.l = np.ones(theta.shape) if lengths is None else lengths
        self.m = np.ones(theta.shape) if masses is None else masses
        self.g = 9.81 # gravity

    def step(self, delta_t, step_size=0.0005):
        for i in range(math.ceil(delta_t / step_size)):
            self.smol_step(step_size)

    def smol_step(self, delta_t):
        m_1, m_2 = self.m[:, 0], self.m[:, 1]
        t_1, t_2 = self.t[:, 0], self.t[:, 1]
        l_1, l_2 = self.l[:, 0], self.l[:, 1]
        t_d1, t_d2 = self.t_d[:, 0], self.t_d[:, 1]
        bottom = (2 * m_1) + m_2 - (m_2 * np.cos(2 * (t_1 - t_2)))
        theta_dd1 = (-self.g * (2 * m_1 + m_2) * np.sin(t_1) - m_2 * self.g * np.sin(t_1 - 2 * t_2) - 2 * np.sin(t_1 - t_2) * m_2 * (t_d2**2 * l_2 + t_d1**2 * l_1 * np.cos(t_1 - t_2))) / (l_1 * bottom)
        theta_dd2 = (2 * np.sin(t_1 - t_2) * (t_d1**2 * l_1 * (m_1 + m_2) + self.g * (m_1 + m_2) * np.cos(t_1) + t_d2**2 * l_2 * m_2 * np.cos(t_1 - t_2))) / (l_2 * bottom)
        self.t_d[:, 0] += delta_t * theta_dd1
        self.t_d[:, 1] += delta_t * theta_dd2
        self.t += delta_t * self.t_d

    def get_nth_pendulum(self, n):
        return self.t[n, 0], self.t[n, 1]


class Physical2Pendulum(Scene):
    def construct(self):
        ps = [0, 3, 0]
        scale = 2.5
        angles = True
        history = defaultdict(lambda: [])
        theta_hist = defaultdict(lambda: [])
        theta1 = MathTex(r"\theta_1")
        theta2 = MathTex(r"\theta_2")

        def mod_clamp(x, low=-math.pi, high=math.pi):
            return ((x-((high-low)/2)) % (high-low)) + low
        
        def lin_scale(x, start, end):
            return (x - start) / (end - start)

        def get_between(start, end, proportion):
            return start + (end - start) * proportion

        def get_pendulum(p, pid, tracers=False, color=BLUE, dot_radius=DEFAULT_DOT_RADIUS):
            t_1, t_2 = p.get_nth_pendulum(0)
            p0 = [ps[0] + scale * math.sin(t_1), ps[1] - scale * math.cos(t_1), 0]
            p1 = [p0[0] + scale * math.sin(t_2), p0[1] - scale * math.cos(t_2), 0]
            x0, x1 = DashedLine([ps[0], ps[1], 0], [ps[0], ps[1] - 1, 0]), DashedLine([p0[0], p0[1], 0], [p0[0], p0[1] - 1, 0])
            l0, l1 = Line(ps, p0), Line(p0, p1)
            x0.set_color(color)
            x1.set_color(color)
            l0.set_color(color)
            l1.set_color(color)

            theta_hist[pid].append(np.array([lin_scale(mod_clamp(t_1), -math.pi, math.pi), lin_scale(mod_clamp(t_2), -math.pi, math.pi), 0]))
            history[pid].append(p1)
            if len(history[pid]) > 10:
                history[pid] = history[pid][1:]

            res = [ l0, l1, Dot(ps, radius=dot_radius, color=WHITE), Dot(p0, radius=dot_radius, color=color), Dot(p1, radius=dot_radius, color=color)]
            if angles:
                flip0, flip1 = p0[0] < ps[0], p1[0] < p0[0]
                res += [ x0, x1, Angle(x0, l0, other_angle=flip0), Angle(x1, l1, other_angle=flip1),
                    MathTex(r"\theta_1").move_to(Angle(x0, l0, other_angle=flip0, radius=0.4 + 3 * SMALL_BUFF).point_from_proportion(0.5)),
                    MathTex(r"\theta_2").move_to(Angle(x1, l1, other_angle=flip1, radius=0.4 + 3 * SMALL_BUFF).point_from_proportion(0.5))]
            if tracers:
                res += [ Line(history[pid][i-1], history[pid][i], color=color) for i in range(1, len(history[pid])) ]
            return res


        def create_pendulum(theta, pid=0, color=BLUE, rate=1, add_updater=True, tracers=True):
            p = DoublePendulum(theta)
            pendulum = VGroup()
            history[pid].clear()
            for obj in get_pendulum(p, pid, color=color, dot_radius=0.06):
                pendulum.add(obj)

            def step(pendulum, dt):
                clear(pendulum)
                p.step(dt * rate)
                for obj in get_pendulum(p, pid, color=color, tracers=tracers, dot_radius=0.06):
                    pendulum.add(obj)

            if add_updater:
                pendulum.add_updater(step)
            return p, pendulum, step

        def clear(*objs):
            for obj in objs:
                for o in obj:
                    obj.remove(o)

        def create_pendulum_path(pid, color, s, e):
            path = VGroup()

            def update(pth, dt):
                clear(pth)
                for obj in get_plot_line(pid, s, e, color):
                    pth.add(obj)
            path.add_updater(update)
            return path, update

        def get_plot_line(pid, domain_start, domain_end, color):
            res = []
            for i in range(1, len(theta_hist[pid])):
                p0 = domain_start + theta_hist[pid][i - 1] * (domain_end - domain_start)
                p1 = domain_start + theta_hist[pid][i] * (domain_end - domain_start)
                if np.linalg.norm(p1 - p0) <= 4:
                    res.append(Line(p0, p1, color=color))
            return res

        def get_color(theta, phi, radius=127, normalize=False):
            return [(127 + radius * math.cos(phi) * math.sin(theta)) / 255, 
                (127 + radius * math.sin(phi) * math.sin(theta)) / 255,
                (127 + radius * math.cos(theta)) / 255] if normalize else \
            [math.floor(127 + radius * math.cos(phi) * math.sin(theta)), 
                math.floor(127 + radius * math.sin(phi) * math.sin(theta)),
                math.floor(127 + radius * math.cos(theta))]

        def get_coloring(grid_size):
            data = [[get_color(get_between(-math.pi, math.pi, i / (grid_size - 1)), get_between(math.pi, -math.pi, j / (grid_size - 1))) for i in range(grid_size)] for j in range(grid_size)]
            return ImageMobject(np.uint8(data))

        def update_color(object, pid, add_updater=True):
            def update(obj, dt):
                if len(theta_hist[pid]) >= 1:
                    obj.set_fill(rgb_to_color(get_color(get_between(-math.pi, math.pi, theta_hist[pid][-1][0]), get_between(-math.pi, math.pi, theta_hist[pid][-1][1]), normalize=True)))
            if add_updater:
                object.add_updater(update)
            return update
        
        def update_circle_pos(obj, pid, s, e, add_updater=True):
            def update(my_obj, dt):
                if len(theta_hist[pid]) >= 1:
                    my_obj.move_to(s + theta_hist[pid][-1] * (e - s))
            if add_updater:
                obj.add_updater(update)
            return update


        text1 = Tex("This is a double pendulum.").move_to([0, -3, 0])
        text2 = Tex("Its behavior is chaotic.").move_to([0, -3, 0])
        text3 = Tex(r"This means that small changes to the initial conditions \\ result in wildly different behaviors.").move_to([0, -3, 0])
        text4 = Tex("Let's see this happen.").move_to([0, -3, 0])
        text5 = Tex(r"Let's do this again, but plot \\ how the angles of each \\ pendulum change over time.").to_corner(LEFT * 0.5 + DOWN)
        text6 = Tex(r"Let's assign a unique color to each point on our plot.").move_to([0, -3.5, 0])
        text7 = Tex(r"The pendulum's state $(\theta_1, \theta_2)$ can be mapped to a color").move_to([0, -3.5, 0])

        p, pendulum, s = create_pendulum(np.ones((1, 2)) * math.pi / 2.0, BLUE, add_updater=False)
        self.add(pendulum)
        self.play( Create(pendulum) )
        self.play( Create(text1) )
        pendulum.add_updater(s)
        self.wait(4)

        clear(pendulum)
        self.play( Transform(text1, text2) )
        self.wait(5)

        clear(pendulum)
        self.play( Transform(text1, text3) )
        self.wait(6)
        pendulum.remove_updater(s)
        self.play(FadeOut(pendulum))
        self.remove(pendulum)

        #######################################################################
        angles = False
        p1, pendulum1, s1 = create_pendulum(np.ones((1, 2)) * math.pi / 2.0, pid=1, color=BLUE, add_updater=False)
        p2, pendulum2, s2 = create_pendulum(np.ones((1, 2)) * math.pi / 1.9, pid=2, color=ORANGE, add_updater=False)
        self.add(pendulum1, pendulum2)
        self.play( Create(pendulum1), Create(pendulum2), Transform(text1, text4) )
        pendulum1.add_updater(s1)
        pendulum2.add_updater(s2)
        self.wait(15)
        pendulum1.remove_updater(s1)
        pendulum2.remove_updater(s2)
        self.play(FadeOut(pendulum1), FadeOut(pendulum2))
        self.remove(pendulum1, pendulum2)

        #######################################################################
        plot = Square(side_length=6.5).to_corner(RIGHT + UP)
        t1, t2 = MathTex(r"\theta_1").shift(3.3611 * RIGHT + 3.5 * DOWN), MathTex(r"\theta_2").move_to([-0.3, 0.25, 0])
        p_pi1, m_pi1 = MathTex(r"\pi").move_to([-0.2, 3.5, 0]), MathTex(r"-\pi").move_to([0.1111, -3.25, 0])
        p_pi2 = MathTex(r"\pi").move_to([6.61111, -3.25, 0])
        self.add(plot, t1, t2)
        self.play( Transform(text1, text5), Create(plot),
            Create(t1), Create(t2),
            Create(p_pi1), Create(m_pi1),
            Create(p_pi2)
        )

        p_start = np.array([0.11111, -3, 0])
        p_end = np.array([6.6111, 3.5, 0])
        ps = [-3.5, 3, 0]
        scale = 1.7
        p1, pendulum1, s1 = create_pendulum(np.ones((1, 2)) * math.pi / 2.0, pid=3, color=BLUE, add_updater=False)
        p2, pendulum2, s2 = create_pendulum(np.ones((1, 2)) * math.pi / 1.9, pid=4, color=ORANGE, add_updater=False)
        path1, p_s1 = create_pendulum_path(3, BLUE, p_start, p_end)
        path2, p_s2 = create_pendulum_path(4, ORANGE, p_start, p_end)
        self.add(pendulum1, pendulum2, path1, path2)
        self.play( Create(pendulum1) )
        self.wait(1)
        pendulum1.add_updater(s1)
        pendulum2.add_updater(s2)
        self.wait(18)
        pendulum1.remove_updater(s1)
        pendulum2.remove_updater(s2)
        path1.remove_updater(p_s1)
        path2.remove_updater(p_s2)
        self.play(FadeOut(pendulum1), FadeOut(pendulum2), FadeOut(path1), FadeOut(path2))
        self.play(Transform(text1, text6), FadeOut(p_pi1), FadeOut(p_pi2), FadeOut(m_pi1), FadeOut(t1), FadeOut(t2))
        self.remove(pendulum1, pendulum2, path1, path2)
        
        #######################################################################
        angles = True
        grid_size = 878
        ps = [-3.5, 3, 0]
        scale = 1.7
        p, pendulum, s = create_pendulum(np.ones((1, 2)) * math.pi / 2.0, pid=5, color=BLUE, add_updater=False, tracers=False)
        coloring = get_coloring(grid_size).scale(270 / grid_size * 6.45 / 2).to_corner(LEFT + UP)
        text_mapping = MathTex(r"(\theta_1, \theta_2) \to ").to_corner(LEFT * 2 + DOWN * 2)
        square_fill = Square(0.5, fill_opacity=1).move_to([-3.5, -2.75, 0])
        square_fill.set_fill(rgb_to_color(get_color(math.pi / 2, math.pi / 2, normalize=True)))
        lens = Circle(radius=0.3, fill_opacity=1, color=WHITE).move_to(p_start + (p_end - p_start) * np.array([lin_scale(mod_clamp(math.pi / 2), -math.pi, math.pi), lin_scale(mod_clamp(math.pi / 2), -math.pi, math.pi), 0]))
        lens.set_fill(rgb_to_color(get_color(math.pi / 2, math.pi / 2, normalize=True)))
        self.wait(1)
        self.add(coloring)
        self.play( coloring.animate.move_to([0.11111 + 6.5 / 2, -3 + 6.5 / 2, 0]))
        self.wait(3)
        self.add(pendulum)
        self.play( Create(pendulum), Transform(text1, text7))
        self.wait(2)
        self.play( Create(text_mapping), Create(square_fill), Create(lens) )
        self.wait(2)
        sf_u = update_color(square_fill, 5, add_updater=True)
        l_u = update_color(lens, 5, add_updater=True)
        l_u2 = update_circle_pos(lens, 5, p_start, p_end, add_updater=True)
        pendulum.add_updater(s)
        self.wait(20)
        pendulum.remove_updater(s)
        lens.remove_updater(l_u)
        lens.remove_updater(l_u2)
        square_fill.remove_updater(sf_u)
        self.play( FadeOut(lens), FadeOut(square_fill), FadeOut(pendulum), FadeOut(coloring), FadeOut(text_mapping), FadeOut(text1), FadeOut(plot))
        self.remove(lens, square_fill, pendulum, coloring, text_mapping, text1, plot)
        self.wait(1)



class FractalPlot(Scene):
    def construct(self):
        grid_size = 945
        angle_domain = [-math.pi, math.pi]
        angle_from_proportion = lambda x: (angle_domain[1] - angle_domain[0]) * x + angle_domain[0]
        theta = np.ones((grid_size**2, 2))
        for i in range(grid_size**2):
            x, y = i % grid_size, i // grid_size
            theta[i, 0] = angle_from_proportion(x / grid_size)
            theta[i, 1] = angle_from_proportion(y / grid_size)
        p = DoublePendulum(theta)

        def get_color(theta, phi, radius=127):
            return [math.floor(127 + radius * math.cos(phi) * math.sin(theta)), 
                math.floor(127 + radius * math.sin(phi) * math.sin(theta)),
                math.floor(127 + radius * math.cos(theta))]

        def get_fractal():
            return np.uint8([[get_color(p.t[j * grid_size + i, 0], p.t[j * grid_size + i, 1]) for i in range(grid_size)] for j in range(grid_size)])

        def step(fractal, dt):
            p.step(dt)
            fractal.pixel_array = np.array(get_fractal())
            fractal.change_to_rgba_array()

        text1 = Tex("Now, lets see how every possible pendulum evolves together.").move_to([0, -3.5, 0])
        text2 = Tex("Every pixel represents the state of a unique pendulum").move_to([0, -3.5, 0])
        text3 = Tex("We can update every pendulum and see how the states change.").move_to([0, -3.5, 0])
        # One unit is 270 px
        fractal = ImageMobject(get_fractal()).scale(270 / grid_size * 7 / 2).to_edge(UP * 0.25)
        self.add(fractal)
        self.add(text1)
        self.wait(5)
        self.play( Transform(text1, text2) )
        self.wait(5)
        self.play( Transform(text1, text3) )
        self.wait(2)
        fractal.add_updater(step)
        self.wait(30)
        fractal.remove_updater(step)
        self.play(FadeOut(fractal), FadeOut(text1))
        self.clear()
        self.wait(3)


if __name__ == "__main__":
    s = "FractalPlot"
    if len(sys.argv) > 1:
        print(' '.join(['manim', *sys.argv[1:], 'pendulum.py', s]))
        subprocess.run(['manim', *sys.argv[1:], 'pendulum.py', s])
    else:
        print(' '.join(['manim', 'pendulum.py', s]))
        subprocess.run(['manim', 'pendulum.py', s])