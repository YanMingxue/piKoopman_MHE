README
===========================

This repo includes the source code of paper [**Self-tuning moving horizon estimation of nonlinear systems via physics-informed machine learning Koopman modeling**](https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.18649 "Go to the paper page on AIChE Journal").

## Component

* [Physics-informed stochastic Koopman modeling](#physics-informed-stochastic-koopman-modeling)
* [Moving horizon estimation](#moving-horizon-estimation)



### Physics-informed stochastic Koopman modeling

A stochastic Koopman model is formulated as follows:

<img src="image/Koopman.png" alt="equation" width="600"/>


### Moving horizon estimation

A moving horizon estimation with automatic generated weighting matrices is designed. The optimization problem is in the following form:

<img src="image/MHE.png" alt="equation" width="600"/>

with the objective function to be   
<img src="image/Obj.png" alt="equation" width="900"/>
and the stage cost
<img src="image/stage_cost.png" alt="equation" width="300"/>

## Model structure

<img src="image/model.png" alt="equation" width="800"/>
