import torch
import math
import random
import pandas as pd
from tqdm import tqdm
import pennylane as qml
from pennylane import numpy as np


n_qubits = 2 # len(X_final[0])
dev_kernel = qml.device("default.qubit", wires=n_qubits, shots=None)
projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel, interface="autograd")
def kernel(x1, x2):
    """The quantum kernel"""
    
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.U1(x1[0], wires=0)
    qml.U1(x1[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.U1(np.pi/((1+np.cos(x1[0]))*(1+np.cos(x1[1]))), wires=1)
    qml.CNOT(wires=[0, 1])
   
    qml.adjoint(qml.CNOT)(wires=[0, 1])
    qml.adjoint(qml.U1)(np.pi/((1+np.cos(x2[0]))*(1+np.cos(x2[1]))), wires=1)
    qml.adjoint(qml.CNOT)(wires=[0, 1])
    qml.adjoint(qml.U1)(x2[1], wires=1)
    qml.adjoint(qml.U1)(x2[0], wires=0)
    qml.adjoint(qml.Hadamard)(wires=1)
    qml.adjoint(qml.Hadamard)(wires=0)
    
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

## Define the quantum kernel matrix 

def kernel_matrix(A, B):
    
    return np.array([[kernel(a, b) for b in B] for a in A])