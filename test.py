from hypothesis_testing import t_test_ind,t_test_1sample
import numpy as np

x1 = np.random.randn(100)
x2 = np.random.randn(100)*2 + 5
t_test_ind(x2,x1,alternative = 'two-sided')
t_test_1sample(x1,0,alternative = 'two-sided')