# waveq

Waveq is a simulation for the 2D Schrodinger equation with arbitrary potential terms. It is based on [taichi](https://docs.taichi-lang.org) and python. To do this, you need `python3.10` and no later and install taichi by

```shell
pip install taichi
```

once this is done, you can run

```shell
python wavefunction.py
```

with the following controls:

```
[SPACE]: change view modes between probability and real/imaginary parts
[ESC]: exit
P: Overlay a visualization of the potential
C: Clear, i.e. set the wavefunction to zero
[MOUSE]: Clicking (and/or dragging) the mouse disturbs the wavefunction by a pulse. 
```

Here are some pictures from this

| ![setup](Media/Oscillator-Setup.gif)              | ![ball](Media/ball.gif)           |
| ------------------------------------------------- | --------------------------------- |
| ![2-sources-complex](media/2-sources-complex.gif) | ![2-sources](media/2-sources.gif) |

