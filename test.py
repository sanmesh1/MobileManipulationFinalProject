#import numpy as np
#a = np.array([1,2,3,4])
#for i in range(a.size):
#    a[i]=a[i]+2
#print(a)

import numpy as np
from scipy.linalg import pinv2
Jwork =  np.array([[-1.99916677e-01, -9.99583385e-02, -5.27355937e-16],
 [ 5.99500104e+00,  3.99750052e+00,  2.00000000e+00],
 [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])

Jfail =  np.array([[-0.39853423, -0.14790343,  0.07717784],
 [ 5.97003849,  3.98580459,  1.99851034],
 [ 1,          1,          1        ]])
#Jwork = np.around(Jwork, decimals=4)
#Jfail = np.around(Jfail, decimals=4)
print("Jwork = ", Jwork)
print("Jfail = ", Jfail)


Jdiff = Jwork - Jfail

print("Jdiff = ", Jdiff)

JworkInv = np.linalg.pinv(Jwork)
JfailInv = np.linalg.pinv(Jfail)
JworkInvScipy = pinv2(Jwork)
JfailInvScipy = pinv2(Jfail)

print("JworkInv = ", JworkInv)
#print("JworkInvScipy = ", JworkInvScipy)
print("JfailInv = ", JfailInv)
#print("JfailInvScipy = ", JfailInvScipy)


