# Code submission by Dhruv Bhatnagar for QOSF assesment Task 2
# Please refer to the attached ReadMe file!
# Run the entire code for optimization results (displayed in a plot).
# pdf (pmf) stands for probability distribution (mass) function

%matplotlib inline
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
# Loading your IBM Q account(s) - Please configure appropriately!
provider = IBMQ.load_account()

# Some hyperparameters/configuration settings
# To control the ansatz
rot_depth = 1
num_seg = 1
y_before_x = True

# To control optimization
g_scaling_factor = 10000
g_cost_fun_order = 5

# To control debugging prints
g_is_quiet = True

# To control the noise model
# Error probability
probabilities = np.zeros([4,4])
probabilities[0] = [0.97, 0.01, 0.01, 0.01]
probabilities[1] = [0.01, 0.97, 0.01, 0.01]
probabilities[2] = [0.01, 0.01, 0.97, 0.01]
probabilities[3] = [0.01, 0.01, 0.01, 0.97]
# readout error
error_1 = noise.ReadoutError(probabilities)

# Add errors to noise model
noise_model = noise.NoiseModel()
noise_model.add_readout_error(error_1, [1,2])
print("Noise model in effect: ", noise_model)
print("***********************************************************************")

# Functions
def ret_prob(param, num_shots, ret_prob, g_draw_ansatz, is_noise_active):
    #creates ansatz, simulates state_vector or the resulting pmf based on ret_prob argument
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(2, 'cr')
    qcirc = QuantumCircuit(qr, cr)
    phi = np.reshape(param, (2, num_seg, rot_depth))
    for i in range(num_seg):
        for j in range(rot_depth):
            if(y_before_x == True):
                qcirc.ry(phi[0][i][j], 0)
                qcirc.rx(phi[1][i][j], 0)
            else:
                qcirc.rx(phi[1][i][j], 0)
                qcirc.ry(phi[0][i][j], 0)
        qcirc.cnot(0,1)
    pmf = np.zeros([4])
    if(g_draw_ansatz == True):
        print("Chosen ansatz for quantum circuit has the following form: ")
        print(qcirc) # can replace by qcirc.draw(output='mpl')
        print("***********************************************************************")
        g_draw_ansatz = False
    if(ret_prob == True):
        # Return probabilities
        qcirc.measure(qr[0], cr[0])
        qcirc.measure(qr[1], cr[1])
        backend_sim = Aer.get_backend('qasm_simulator')
        if(is_noise_active == True):
            # with noise
            job_sim = execute(qcirc, backend_sim, shots=num_shots, noise_model = noise_model)
        else:
            job_sim = execute(qcirc, backend_sim, shots=num_shots)
        result_sim = job_sim.result()
        count = result_sim.get_counts(qcirc)
        try:
            pmf[0] = count["00"]
        except:
            pass
        try:
            pmf[1] = count["01"]
        except:
            pass
        try:
            pmf[2] = count["10"]
        except:
            pass
        try:
            pmf[3] = count["11"]
        except:
            pass
        if(g_is_quiet == False):
            print("calculated pdf = ", pmf/num_shots)
            print("***********************************************************************")
        return pmf/num_shots # gives the pdf
    else:
        # return statevector
        # statevector is not used in optimization, so it is calculated without noise.
        simulator_sv = Aer.get_backend('statevector_simulator')
        result_sv = execute(qcirc, simulator_sv).result()
        statevector_sv = result_sv.get_statevector(qcirc)
        print("statevector of optimized circuit = ", statevector_sv)
        print("pdf of output state = ", np.absolute(statevector_sv)*np.absolute(statevector_sv))
        print("***********************************************************************")
        return pmf
        
def cost_fun(input_pdf):
    ref_pdf = np.array([0.5, 0, 0, 0.5])
    cost = np.linalg.norm((input_pdf - ref_pdf), ord=g_cost_fun_order) / g_scaling_factor
    if(g_is_quiet == False):
        print("unscaled cost =", cost * g_scaling_factor)
        print("***********************************************************************")
    return cost

def fn_to_opt(phi0):
    return cost_fun(ret_prob(phi0, num_shots_g, True, False, True))

def primary_func():
    # Initial values of parameters
    param0 = 180*np.random.rand(2 * num_seg * rot_depth)
    print("Initial parameters = ", param0)
    print("***********************************************************************")
    res = minimize(fn_to_opt, param0, method='Powell', options={'disp':True})
    print("Results: Parameters obtained after optimization = ", res.x)
    print("***********************************************************************")
    y = ret_prob(res.x, num_shots_g, False, False, False)
    print("Number of shots = ", num_shots_g, "Final value of unscaled cost function = ", g_scaling_factor*cost_fun(ret_prob(res.x, num_shots_g, True, False, False)))
    print("***********************************************************************")
    return g_scaling_factor*cost_fun(ret_prob(res.x, num_shots_g, True, True, False))

# Execution 
cost_arr = np.zeros(5)
num_shots_arr = np.array([1,10,100,1000,10000])
num_shots_g = 1
cost_arr[0] = primary_func()
num_shots_g = 10
cost_arr[1] = primary_func()
num_shots_g = 100
cost_arr[2] = primary_func()
num_shots_g = 1000
cost_arr[3] = primary_func()
num_shots_g = 10000
cost_arr[4] = primary_func()

# Results
plt.figure(num=None, figsize=(14, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(num_shots_arr, cost_arr, linestyle='-', marker='o', markersize=11, markerfacecolor='blue', color='purple') 
plt.xlabel('Number of shots') 
plt.ylabel('Final cost function after optimization') 
plt.title('Cost vs Number of shots') 
plt.grid(True)
plt.show()

# EOF
