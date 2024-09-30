## Kolmogorov Arnold Network

This is a simplified implementation of a Kolmogorov Arnold network.
The simplification lies in the fact that the learnable functions *f* are constructed like this:  

Simple_Kan :*f(x) = spline(x)*

Original kan paper :*f(x) = ws spline(x) + wb silu(x)*.

## Credits

link to the original paper:https://arxiv.org/abs/2404.19756  
link to the paper's implementation:https://github.com/KindXiaoming/pykan

## Toy Example

!(https://github.com/StavrosNe/Neural-Nets/blob/main/Kolmogorov%20Arnold%20Network/example2.png "Toy Exmple")
