# cmenpy

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl-3.0)
![PyPI - Version](https://img.shields.io/pypi/v/cmenpy?style=flat-square)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/cmenpy?style=flat-square)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cmenpy?style=flat-square)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/cmenpy?style=flat-square)
![GitHub Release Date](https://img.shields.io/github/release-date/ltsim/cmenpy.svg?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cmenpy?style=flat-square)

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/publish.yml?style=flat-square&logo=pypi&label=Publish)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/test.yml?style=flat-square&logo=pytest&label=Testing)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ltsim/cmenpy/type.yml?style=flat-square&logo=python&label=Typed)

**C**ompute **M**etaheuristic **En**gine for **P**ython (_cmenpy_ or _/siːmɛnpaɪ/_) is an extensible, flexible and vectorized Python framework explicitly architected for designing and prototyping metaheuristic and population-based optimization algorithms.

It provides the clean, abstractions necessary to build custom evolutionary algorithms, swarm intelligence methods, and hybrid heuristics from scratch, granting you absolute architectural control.

## Core Pillars

* **Vectorized Data Structures:** Built from the ground up to maximize data locality and mathematical efficiency. Population arrays, agent states, and search spaces leverage a completely flattened, vectorized structure that eliminates heavy object overhead and interfaces natively with NumPy and parallel computing paradigms.
* **Extensible Architecture:** Designed with clean, highly pedagogical abstractions. Crafting custom transition operators, custom mutation rules, or completely new optimization paradigms requires overriding just a few decoupled, well-defined methods.
* **Non-Monolithic Flexibility:** We cut out all the bloat. Bypasses secondary overhead—such as visualization tools, complex file exporters, and rigid internal loggers—focusing entirely on the raw mathematical engine and structural flow.
* **Research Ready:** Protected by the **GNU GPLv3 License**. You have complete freedom to design, modify, and distribute your custom algorithms. Any derivative work or integration must also remain open and free, ensuring the software permanently evolves as a public good for both academic research and open commercial development.

---

* Developed by: [LTSIM](mailto:tsim@cucei.udg.mx) @ 2026