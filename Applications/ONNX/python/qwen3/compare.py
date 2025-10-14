import numpy as np

arr1 = np.fromfile("./modelling_logits.bin",dtype="float32").reshape(1,151936)
arr2 = np.fromfile("../../jni/nntrainer_logits.bin",dtype="float32").reshape(1,151936)

if(np.allclose(arr1,arr2,atol=1e-4,rtol=1e-4)):
    print("equal")
else:
    print("not equal")    