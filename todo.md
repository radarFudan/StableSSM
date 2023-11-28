Plan:

0. Add the synthetic task of linear functional to the dataset.
   Check the code for LF and CIFAR100
1. This is going to be the code release for my StableSSM draft.
2. Collect the state of the art implementation of linear RNN first.
3. By FFT (Seems to be some memory issue - https://github.com/pytorch/pytorch/issues/94893)
4. Might refer to the FlashFFTConv for the speedup. (https://github.com/HazyResearch/flash-fft-conv)
5. By Associative Scan (Currently there is an implementation that somehow need jax)
6. Prepare some simple test codes so that I know the growth and constant coefficient in terms of the time and space cost.
7. Implement my stable parameterization individually
8. Can we construct a hypernetwork and also train it by gradient descent with the constraint from my stability criterion for stable approximation?

Environment preparation:
\`\`
