Plan:
1. This is going to be the code release for my StableSSM draft.
2. Collect the state of the art implementation of linear RNN first.
  1. By FFT (Seems to be some memory issue - https://github.com/pytorch/pytorch/issues/94893)
  2. Might refer to the FlashFFTConv for the speedup. (https://github.com/HazyResearch/flash-fft-conv)
  4. By Associative Scan (Currently there is an implementation that somehow need jax)
  5. Prepare some simple test codes so that I know the growth and constant coefficient in terms of the time and space cost.
3. Implement my stable parameterization individually
4. Can we construct a hypernetwork and also train it by gradient descent with the constraint from my stability criterion for stable approximation?
