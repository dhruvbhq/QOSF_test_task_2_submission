# QOSF_test_task_2_submission
This is Dhruv Bhatnagar's submission to the QOSF assessment task no. 2.

*Update - August 2020*

*As per the results of the QC mentorship program (QOSF) opening tasks, this submission was among the 56 to be accepted into the program and to have finished the task that the organization asked for, from almost 200 people that applied.*

*The problem statement for task 2 was to implement a circuit which returns a given quantum state with certain requirements, such as the type of gates allowed were fixed, and optimizing to find the right set of parameters for the task.*

*\<End of update\>*


Some information to interpret the code:
1. The code is based on IBM's qiskit library - please configure your setup/credentials correctly to run it.
2. The approach used is to take an ansatz for the quantum circuit and to optimize the output a cost function.
3. The cost function calculates the difference between the probability mass function (pmf) of the circuit and the desired pmf.
4. The parameters over which optimization is carried out are the angles of rotation for the quantum gates involved.
5. The ansatz is highly configurable using the following 'hyper parameters':
  5a. rot_depth: This controls the number of rotation gates (Rx and Ry on the first qubit) in each segment of the circuit.
  5b. num_seg: A segment consists of a unit of Rx and Ry gates in alternating sequence, followed by a CNOT. num_seg controls the number of   segments that exist in the ansatz.
  5c. y_before_x: If this is true, the ansatz is created by applying Ry gates before Rx gates.
6. The ansatz circuit can be drawn for visualization by passing an argument g_draw_ansatz as True in the function ret_prob.
7. The optimization can also be configured as follows:
  7a. g_scaling_factor scales the cost function used for optimization.
  7b. g_cost_fun_order sets the order of vector norm used in the cost function.
  7c. These were introduced to improve the final cost function values.
8. A ReadOut error model has been incorporated in the circuit, with a given error probability matrix.
9. The ret_prob function takes as input the parametr values, creates ansatz, and simulates state_vector or the resulting pmf based on ret_prob argument.
10. The cost_fun function calculates the cost function.
11. The fn_to_opt function takes only the parameters as input and returns the cost.
12. The primary_func function initializes initial oarameters randomly, optimizes and prints the final statevector and the cost.
13. The 'Powell' method of optimization was chosen as it does not require derivatives.
14. primary_func is then called for several values of num_shots, the number of shots of measurement, and the resulting cost values after optimization are plotted. In general, the cost decreases with the number of shots.
