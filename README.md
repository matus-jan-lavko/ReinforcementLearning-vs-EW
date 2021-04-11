# ReinforcementLearning-vs-EW

This is a repository of my Bachelor's thesis project Reinforcement Learning vs. 1/n and Mean-Variance Optimization In The Portfolio Allocation Problem. In this project I am testing a number of model-free Reinforcement Learning algorithms based on the framework from https://arxiv.org/abs/2011.09607 written on top of OPENAI stable-baselines. Using data from Bloomberg of the broad equity U.S and EU listed securities I am trying to answer the main research question:

**Based on price data and features constructed wherewith, do model-free RL algorithms outperform the Occam's Razor 1/n portfolio and the different linear specifications optimized using a convex objective over a convex set of constraints?**

## Sample tests:

<img src="https://user-images.githubusercontent.com/69042407/114295827-f248b800-9aa7-11eb-9f4a-8c0d16633688.gif" float = none width="300" height="270">
<img src="https://user-images.githubusercontent.com/69042407/114295923-aea27e00-9aa8-11eb-9f90-c406f7169d34.gif" float = none width="300" height="270">
<img src="https://user-images.githubusercontent.com/69042407/114295934-c24de480-9aa8-11eb-8558-3ce8511e8694.gif" float = none width="300" height="270">
<img src="https://user-images.githubusercontent.com/69042407/114295953-d560b480-9aa8-11eb-9805-ddbe2347b917.gif" float = none width="300" height="270">
<img src="https://user-images.githubusercontent.com/69042407/114295954-d72a7800-9aa8-11eb-8d97-0d65e9bd66a2.gif" float = none width="300" height="270">

## Dependencies

The project and the notebooks have been developed on **python = 3.6.0**. All of the dependencies can be installed using:

```
pip install -r dependencies.txt
```


