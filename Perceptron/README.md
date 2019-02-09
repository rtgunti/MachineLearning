Python program that implements a single perceptron using the delta rule presented in
the lecture. Use the following activation function:
^y =
(
1 if ~wT ~x > 0
0 else
where ~w is the vector of weights including the bias (w0). Treat all attributes and weights
as double-precision values.
Given are the two data sets1 named Example and Gauss as tsv (tabular separated values)
les. Your program should be able to read both data sets and treat the rst value of each
line as the class (A or B). In order to get the same results, class A is to be treated as the
positive class, hence y = 1, and class B as the negative one (y = 0). All weights are to
be initialized with 0. Your task is to correctly implement the perceptron learning rule in
batch mode with a constant (t = 0) and an annealing (t = 0
t ) learning rate (in both
cases 0 = 1), i.e:
~wt+1   ~wt +
X
~x2Y(~x; ~w)
t(y 􀀀 ^y)~x
