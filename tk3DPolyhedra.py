#!/usr/bin/env python3

"""
tk3DPolyhedra.py

Author:      J. van Oostrum
Date:        2021-05-30
Description: Rotating polyhedra using a 'standard' tk canvas.
             Flat shading and wireframe mode.

The (pythonized) Tcl/Tk code is based on polyhedra.tcl (2005) by Gerard Sookahet
Tcl wiki: https://wiki.tcl-lang.org/page/3D+polyhedra+with+simple+tk+canvas
"""

import tkinter as tk
import numpy as np
from math import pi, sin, cos, sqrt
from enum import Enum


class Direction(Enum):
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2


class Gui:
    def __init__(self, root):
        self.root = root
        self.width = 400
        self.height = 400

        # the regular Platonic solids
        self.polygons = [Tetrahedron(), Cube(), Octahedron(), Dodecahedron(), Icosahedron()]
        self.ipoly = 0
        self.poly = self.polygons[self.ipoly]

        self.animate = True
        self.frames = 0
        self.frames_change = 50
        self.angle = [0.0, 0.0, 0.0]  # rotation angle (radians) over the x, y, and z axis
        self.angle_incr = [pi / 120, pi / 60, pi / 30]
        self.angle_step = pi / 360  # angle increment steps (radians)

        self.distance = tk.IntVar()  # the view distance
        self.distance_min = 2500
        self.distance_max = 200

        self.speed = tk.IntVar()  # animation interval (ms)
        self.speed_min = 125
        self.speed_max = 10
        self.wire_clr = 'gray'  # wireframe edge color
        self.text_clr = 'white'
        self.shaded = True

        self.label = tk.Label(root, text="Platonic solids")
        # self.label.pack()

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.configure(state=tk.DISABLED, bg='gray25')
        self.canvas.pack(fill='both', padx='2m', pady='2m')

        # buttons
        b1 = tk.Button(root, text="Wire", command=self.wire_click)
        b1.bind('<Return>', self.wire_click)
        b1.pack(fill=tk.NONE, side=tk.LEFT)

        b2 = tk.Button(root, text="Shade", command=self.shade_click)
        b2.bind('<Return>', self.shade_click)
        b2.pack(fill=tk.NONE, side=tk.LEFT)

        b3 = tk.Button(root, text="Animate", command=self.animate_click)
        b3.bind('<Return>', self.animate)
        b3.pack(fill=tk.NONE, side=tk.LEFT)

        # sliders
        s1 = tk.Scale(root, from_=self.distance_min, to=self.distance_max, label='Distance', showvalue=tk.FALSE, orient=tk.HORIZONTAL, variable=self.distance, command=self.scale_changed)
        s1.set((self.distance_min + self.distance_max) / 2)
        s1.pack(fill=tk.NONE, side=tk.LEFT)

        s2 = tk.Scale(root, from_=self.speed_min, to=self.speed_max, label='Speed', showvalue=tk.FALSE, orient=tk.HORIZONTAL, variable=self.speed, command=self.speed_changed)
        s2.set((self.speed_min + self.speed_max) / 2)
        s2.pack(fill=tk.NONE, side=tk.LEFT)

        self.root.bind('<Control-q>', self.quit_click)

        # pan and zoom stuff
        self.pan_x = 0
        self.pan_y = 0
        self.pan_x_start = 0
        self.pan_y_start = 0

        self.canvas.bind("<2>", self.mouse_pan_start)
        self.canvas.bind("<B2-Motion>", self.mouse_pan)

        self.canvas.bind("<3>", self.mouse_pan_start)
        self.canvas.bind("<B3-Motion>", self.mouse_pan)

        self.display_poly()

    def wire_click(self):
        self.shaded = False
        self.display_poly(False)

    def shade_click(self):
        self.shaded = True
        self.display_poly(False)

    def scale_changed(self, event):
        self.display_poly(False)

    def speed_changed(self, event):
        self.display_poly(False)

    def animate_click(self):
        self.animate = not self.animate
        if self.animate:
            self.display_poly()

    def mouse_pan_start(self, event):
        self.pan_x_start = event.x
        self.pan_y_start = event.y
        self.angle_incr = [0.0, 0.0, 0.0]

    def mouse_pan(self, event):
        dx = event.x - self.pan_x_start
        dy = event.y - self.pan_y_start
        diff = 5
        if abs(dx) > diff and abs(dy) > diff:
            self.angle_incr[2] += np.sign(dx) * self.angle_step
        elif abs(dx) > diff:
            self.angle_incr[0] += np.sign(dx) * self.angle_step
        elif abs(dy) > diff:
            self.angle_incr[1] += np.sign(dy) * self.angle_step
        self.angle = np.add(self.angle, self.angle_incr)
        self.display_poly(False)

    def quit_click(self, event):
        self.root.destroy()

    def projection_2d(self, vertices):
        pvertices = []
        d = self.distance.get()
        for vertex in vertices:
            vx, vy, vz = vertex
            vz = (vz + 10) / d
            # tk canvas upper left corner (x,y) = (0,0)
            # https://stackoverflow.com/questions/7811263/how-to-convert-tkinter-canvas-coordinate-to-window
            pvertices.append([vx / vz + self.width / 2, vy / vz + self.height / 2])
        return pvertices

    def display_poly(self, refresh=True):
        self.display_wireframe()
        if self.animate and refresh:
            self.frames += 1
            if self.frames > self.frames_change:
                self.frames = 0
                self.ipoly += 1
                if self.ipoly >= len(self.polygons):
                    self.ipoly = 0
                self.poly = self.polygons[self.ipoly]
            self.angle = np.add(self.angle, self.angle_incr)
            self.root.after(self.speed.get(), self.display_poly)

    def display_wireframe(self):
        self.canvas.delete('all')  # clear the canvas

        # what poly type do we have?
        text = self.canvas.create_text(10, 10, text=self.poly.whoami(), fill=self.text_clr, font='Helvetica 15')
        self.canvas.itemconfigure(text, state=tk.DISABLED, anchor=tk.NW)

        if self.shaded:
            outline = ''  # no outline color
        else:
            outline = self.wire_clr

        for face in self.poly.faces:
            face_vertices = []
            for i in face:
                face_vertices.append(self.poly.vertices[i])
            vertices = transformation(face_vertices, self.angle)
            normal = surface_normal(vertices)

            # backface ceiling for hidden face. Not removed but reduced to a point.
            if normal[2] < 0.0:  # normal_vector.z is negative
                if self.shaded:
                    grad = (100 + (154 * intensity(normal) / 32))  # color gradation
                    color = "#0000{:02x}".format(round(grad))
                else:
                    color = ''  # no color (wireframe)
                pvertices = self.projection_2d(vertices)
                self.canvas.create_polygon(pvertices, fill=color, outline=outline)
            else:
                b = barycenter(vertices)
                b[0] += self.width / 2
                b[1] += self.height / 2
                bb = [b[0:2] for i in range(len(face))]  # we only need x and y from the barycenter coordinates
                self.canvas.create_polygon(bb, fill='', outline=outline)

        self.canvas.update()

    def display_shade(self):
        self.canvas.update()


class Poly:
    def __init__(self):
        self.vertices = np.array([])
        self.faces = np.array([])

    def whoami(self):
        return(str(self.__class__.__name__))

    def str_normal_vectors(self):
        nvs = ""
        for face in self.faces:
            face_vertices = []
            for i in face:
                face_vertices.append(self.vertices[i])
            face_normal = surface_normal(face_vertices)
            nvs += np.array_str(face_normal)
            nvs += "\n"
        return nvs

    def str_vertices(self):
        vs = ''
        for face in self.faces:
            face_vertices = []
            for i in face:
                face_vertices.append(self.vertices[i])
            vs += ''.join(str(face_vertices))
            vs += "\n"
        return vs


class Plane(Poly):
    def __init__(self):
        super().__init__()
        a = 1
        self.vertices = np.array([[-a, -a, 0],
                                  [a, -a, 0],
                                  [a, a, 0],
                                  [-a, a, 0]])
        self.faces = np.array([[0, 1, 2, 3]])


class Tetrahedron(Poly):
    def __init__(self):
        super().__init__()
        a = 1 / sqrt(3)
        self.vertices = np.array([[a, a, a],
                                  [a, -a, -a],
                                  [-a, a, -a],
                                  [-a, -a, a]])
        self.faces = np.array([[0, 3, 1], [2, 0, 1], [3, 0, 2], [1, 3, 2]])


class Cube(Poly):
    def __init__(self):
        super().__init__()
        a = 1 / 2
        self.vertices = np.array([[a, a, a],
                                  [-a, a, a],
                                  [-a, -a, a],
                                  [a, -a, a],
                                  [a, a, -a],
                                  [-a, a, -a],
                                  [-a, -a, -a],
                                  [a, -a, -a]])
        self.faces = np.array([[4, 7, 6, 5], [0, 1, 2, 3], [3, 2, 6, 7],
                               [4, 5, 1, 0], [0, 3, 7, 4], [5, 6, 2, 1]])


class Octahedron(Poly):
    def __init__(self):
        super().__init__()
        a = 1
        self.vertices = np.array([[a, 0, 0],
                                  [0, a, 0],
                                  [-a, 0, 0],
                                  [0, -a, 0],
                                  [0, 0, a],
                                  [0, 0, -a]])
        self.faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
                               [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5]])


# http://www.sacred-geometry.es/?q=en/content/phi-sacred-solids
class Dodecahedron_ALT(Poly):
    def __init__(self):
        super().__init__()
        phi = (1 + sqrt(5)) / 2
        self.vertices = np.array([[phi, 0, 1 / phi],
                                  [-phi, 0, 1 / phi],
                                  [-phi, 0, -1 / phi],
                                  [phi, 0, -1 / phi],
                                  [1 / phi, phi, 0],
                                  [1 / phi, -phi, 0],
                                  [-1 / phi, -phi, 0],
                                  [-1 / phi, phi, 0],
                                  [0, 1 / phi, phi],
                                  [0, 1 / phi, -phi],
                                  [0, -1 / phi, -phi],
                                  [0, -1 / phi, phi],
                                  [1, 1, 1],
                                  [1, -1, 1],
                                  [-1, -1, 1],
                                  [-1, 1, 1],
                                  [-1, 1, -1],
                                  [1, 1, -1],
                                  [1, -1, -1],
                                  [-1, -1, -1]])
        a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t = (i for i in range(20))
        self.faces = np.array([[n, f, s, d, a], [m, i, l, n, a], [d, r, e, m, a], [e, h, p, i, m],
                               [r, j, q, h, e], [d, s, k, j, r], [p, b, o, l, i], [o, g, f, n, l],
                               [g, t, k, s, f], [q, j, k, t, c], [h, q, c, b, p], [b, c, t, g, o]])


class Dodecahedron(Poly):
    def __init__(self):
        super().__init__()
        a = sqrt(2 / (3 + sqrt(5))) / sqrt(3)
        b = (1 + sqrt(6 / (3 + sqrt(5)) - 2 + 2 * sqrt(2 / (3 + sqrt(5))))) / sqrt(3)
        c = 1 / sqrt(3)
        self.vertices = np.array([[-a, 0, b],
                                  [a, 0, b],
                                  [-c, -c, -c],
                                  [-c, -c, c],
                                  [-c, c, -c],
                                  [-c, c, c],
                                  [c, -c, -c],
                                  [c, -c, c],
                                  [c, c, -c],
                                  [c, c, c],
                                  [b, a, 0],
                                  [b, -a, 0],
                                  [-b, a, 0],
                                  [-b, -a, 0],
                                  [-a, 0, -b],
                                  [a, 0, -b],
                                  [0, b, a],
                                  [0, b, -a],
                                  [0, -b, a],
                                  [0, -b, -a]])
        self.faces = np.array([[0, 1, 9, 16, 5], [1, 0, 3, 18, 7], [1, 7, 11, 10, 9], [11, 7, 18, 19, 6],
                              [8, 17, 16, 9, 10], [2, 14, 15, 6, 19], [2, 13, 12, 4, 14], [2, 19, 18, 3, 13],
                              [3, 0, 5, 12, 13], [6, 15, 8, 10, 11], [4, 17, 8, 15, 14], [4, 12, 5, 16, 17]])


class Icosahedron(Poly):
    def __init__(self):
        super().__init__()
        a = 0.525731112119133606
        b = 0.850650808352039932
        self.vertices = np.array([[-a, 0, b], [a, 0, b], [-a, 0, -b],
                                  [a, 0, -b], [0, b, a], [0, b, -a],
                                  [0, -b, a], [0, -b, -a], [b, a, 0],
                                  [-b, a, 0], [b, -a, 0], [-b, -a, 0]])
        self.faces = np.array([[4, 0, 1], [9, 0, 4], [5, 9, 4], [5, 4, 8],
                               [8, 4, 1], [10, 8, 1], [3, 8, 10], [3, 5, 8],
                               [2, 5, 3], [7, 2, 3], [10, 7, 3], [6, 7, 10],
                               [11, 7, 6], [0, 11, 6], [1, 0, 6], [1, 6, 10],
                               [0, 9, 11], [11, 9, 2], [2, 9, 5], [2, 7, 11]])


def surface_normal(vertices):
    # return surface_normal_newell(vertices)
    return surface_normal_org(vertices)


# https://stackoverflow.com/questions/39001642/calculating-surface-normal-in-python-using-newells-method
def surface_normal_newell(v):
    vertices = np.array(v)
    n = np.cross(vertices[1, :] - vertices[0, :], vertices[2, :] - vertices[0, :])
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError('Zero surface normal vector')
    else:
        normal_vector = n / norm
    return normal_vector


def surface_normal_org(vertices):
    # the (face) plane orientation is defined by two vectors
    x1 = vertices[1][0] - vertices[0][0]
    y1 = vertices[1][1] - vertices[0][1]
    z1 = vertices[1][2] - vertices[0][2]
    v1 = np.array([x1, y1, z1])

    x2 = vertices[2][0] - vertices[1][0]
    y2 = vertices[2][1] - vertices[1][1]
    z2 = vertices[2][2] - vertices[1][2]
    v2 = np.array([x2, y2, z2])

    normal_vector = np.cross(v1, v2)
    return normal_vector


def barycenter(vertices):
    x = y = z = 0
    for vertex in vertices:
        vx, vy, vz = vertex
        x += vx
        y += vy
        z += vz
    x /= len(vertices)
    y /= len(vertices)
    z /= len(vertices)
    return [x, y, z]


def rotation_matrix(phix, phiy, phiz):
    """Return the transformation matrix for a rotation, with the given angles,
    over the x, y, and z axis."""
    mx = np.array([[1, 0, 0],
                   [0, cos(phix), -sin(phix)],
                   [0, sin(phix), cos(phix)]])
    my = np.array([[cos(phiy), 0, sin(phiy)],
                   [0, 1, 0],
                   [-sin(phiy), 0, cos(phiy)]])
    mz = np.array([[cos(phiz), -sin(phiz), 0],
                   [sin(phiz), cos(phiz), 0],
                   [0, 0, 1]])
    return np.dot(mz, np.dot(my, mx))


def transformation(vertices, angle=[0.0, 0.0, 0.0]):
    """Apply transformation to the vertex coordinates."""
    m = rotation_matrix(angle[0], angle[1], angle[2])
    nvertices = np.inner(vertices, m)
    return nvertices


def intensity(surface_normal, light=[1, 1, -1]):
    """Compute the surface color intensity."""
    magnitude = np.inner(surface_normal, surface_normal)
    v = np.inner(light, light)
    a = np.inner(surface_normal, light)
    b = sqrt(magnitude * v)
    intensity = round(31 * a / b)
    if intensity < 0:
        intensity = 31
    else:
        intensity = 32 - intensity
    return intensity


def run():
    root = tk.Tk()
    root.title("Polyhedra")
    Gui(root)
    root.mainloop()


if __name__ == "__main__":
    run()
