import taichi as ti
from taichi.math import exp, cos, sin, pi
ti.reset()
ti.init(arch = ti.gpu, fast_math=True)

# Number of pixels in our grid
n       = 800
dx      = 1/n
dt      = 2e-1 * (2*dx*dx)
V_char  = 2e5
wave    = ti.Vector.field(2, ti.f32, (n,n))
wavenew = ti.Vector.field(2, ti.f32, (n,n))
pixels  = ti.Vector.field(3, ti.f32, (n,n))
V       = ti.field(ti.f32, (n,n))
A       = ti.field(dtype=ti.f32, shape=())
gui     = ti.GUI("2D Waves", res = n, fast_gui=False)
window  = ti.ui.Window("2D Waves", res=(n, n), fps_limit=400)
gui = window.get_canvas()


@ti.kernel
def fill_V(V:ti.template(),n:int):
    for x,y in V:
        X = (2*x-n)/n
        Y = (2*y-n)/n
        # if X**2 + Y**2 <(0.3)**2: V[x,y] = -V_char
        V[x,y] = V_char*(X**2 + Y**2)

@ti.kernel
def initialize(x:float, y:float, px:float, py:float, sx:float, sy:float):
    for i,j in ti.ndrange((1, n-1), (1, n-1)):
        psi = exp(-((2*i-n)/n - x)**2/(2*sx**2) - ((2*j-n)/n - y)**2/(2*sy**2))/(2*pi*sx*sy)
        wave[i,j][0] = psi * cos((2*i-n)/n*px + (2*j-n)/n*py)
        wave[i,j][1] = psi * sin((2*i-n)/n*px + (2*j-n)/n*py)
    
    h = dt/(4*dx*dx)
    for i,j in wave:
        if i!=0 and j!=0 and i!=n-1 and j!=n-1: 
            wave[i,j][1] += (- 4*h - dt*V[i,j]/2)*wave[i,j][0] + h*(wave[i+1,j][0] + wave[i-1,j][0] + wave[i,j+1][0] + wave[i,j-1][0])


@ti.kernel
def add_pulse(x:ti.f32,y:ti.f32,sx:ti.f32,sy:ti.f32,p:ti.f32):
    C = 10000
    for i,j in ti.ndrange((1, n-1), (1, n-1)):
        X = (2*i-n)/n - x
        Y = (2*j-n)/n - y
        R = (X**2 + Y**2)**0.5 
        psi = C*dt*exp(-X**2/(2*sx**2) - Y**2/(2*sy**2))/(2*pi*sx*sy)
        wave[i,j][0] += psi * cos(p*R)
        wave[i,j][1] += psi * sin(p*R)
        wave[i,j]/= (1+C*dt)


@ti.kernel
def draw(n:int, dt:float, dx:float, color:bool, potential:bool, brightness:float):
    h = dt/(2*dx*dx)
    for i,j in ti.ndrange((1, n-1), (1, n-1)):
        wavenew[i,j][0] = wave[i,j][0] - (- 4*h - dt*V[i,j])*wave[i,j][1] - h*(wave[i+1,j][1] + wave[i-1,j][1] + wave[i,j+1][1] + wave[i,j-1][1])
    
    for i,j in ti.ndrange((1, n-1), (1, n-1)):
        wavenew[i,j][1] = wave[i,j][1] + (- 4*h - dt*V[i,j])*wavenew[i,j][0] + h*(wavenew[i+1,j][0] + wavenew[i-1,j][0] + wavenew[i,j+1][0] + wavenew[i,j-1][0])

    # for i,j in pixels:
        pixels[i,j] = ((1-color)*(wavenew[i,j][0]**2 + wave[i,j][1]*wavenew[i,j][1]) + ti.Vector([wavenew[i,j][0], 0, wavenew[i,j][1]],dt=ti.f32)*color)/brightness + potential*ti.Vector([54, 14, 97])*ti.abs(V[i,j]/V_char)/255 
        wave[i,j]   = wavenew[i,j]


if __name__ == "__main__":
    v       = 100
    s       = 1e-1
    color   = False
    potent  = False
    held_P  = False
    held_S  = False
    held_R  = False
    bright  = 1
    brightC = 0.1

    initialize(0.6,0,-v,0,s,s)
    fill_V(V,n)

    while window.running:
        if window.is_pressed(ti.GUI.LMB):
            x,y = window.get_cursor_pos()
            add_pulse((2*x-1), (2*y-1), s, s, v)

        if window.is_pressed(ti.GUI.ESCAPE): break
        if window.is_pressed("c"): wave.fill(0)

        if held_S and not window.is_pressed(ti.GUI.SPACE):
            held_S  = False
            color   = not color
        
        if held_P and not window.is_pressed('p'):
            held_P  = False
            potent  = not potent

        if held_R and not window.is_pressed('r'):
            held_R  = False
            initialize(0.6,0,-v,0,s,s)
        
        if window.is_pressed(ti.GUI.UP):
            if bright>1-brightC: bright -= brightC

        if window.is_pressed(ti.GUI.DOWN):
            if bright<100: bright += brightC

        held_S = window.is_pressed(ti.GUI.SPACE)
        held_P = window.is_pressed('p')
        held_R = window.is_pressed('r')

        draw(n,dt,dx,color,potent,bright)

        gui.set_image(pixels)
        window.show()
