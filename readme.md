# Jigsaw Solver

[Sample shuffled puzzle. Contains two images](images/8_by_11_shuffled.jpg)


[Solution discovered by the solver.](images/8_by_11_solved.jpg)


This is a WIP refactored version of my third year University project.

It aims to solve shapeless jigsaw puzzles without knowing the dimensions. Thus it can solve multiple images at the same time as well as cropped up puzzles.

At the time it was a research project and code quality wasn't really a concern. You can see some pretty dark patterns in the code that I'm working on fixing. However the maths holds up and the solver is reliable.

See the original at [fyp-jigsaw](https://github.com/jeanmalod/fyp-jigsaw)
# Install

```

git clone https://github.com/jeanmalod/jigsaw-solver
```

```
cd fyp-jigsaw
```

```
pip install pipenv && pipenv install
```

```
pipenv shell && python gui.py
```

# Examples

