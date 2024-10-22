# Jigsaw Solver

This is a WIP refactored version of my third year University project.

It aims to solve shapeless jigsaw puzzles without knowing the dimensions. Thus it can solve multiple images at the same time as well as cropped up puzzles.

At the time it was a research project and code quality wasn't really a concern. You can see some pretty dark patterns in the code that I'm working on fixing. However the maths holds up and the solver is reliable.

Read more about how it works and the theory behind it [here](https://jean-malo.com/files/jigsaw-solver.pdf).

See the original code at [fyp-jigsaw](https://github.com/jeanmalod/fyp-jigsaw).

# Example

An 88 pieces puzzle solved by this program, it contains two images.

![Sample shuffled puzzle. Contains two images](images/8_by_11_shuffled.jpg)


![Solution discovered by the solver.](images/8_by_11_solved.jpg)

# Install

```shell

git clone https://github.com/jeanmalod/jigsaw-solver

cd fyp-jigsaw

make download-poetry && make install

poetry shell && python gui.py
```
