# simulated experiments on estimating Floquet quasi-energies with stochastic noise
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, execute,QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram, array_to_latex
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction

from qiskit.circuit.library import QFT
from numpy import pi
from scipy.signal import find_peaks
from time import time

import warnings
warnings.filterwarnings('ignore')


class myqc(QuantumCircuit):
    def set_ct(self,cq,tq):
        self.cq=cq
        self.tq=tq

    # define Pauli gates    
    def paulis(self,qi,x):  
        if x==[0,0]:
            self.id(qi)
                     
        elif x==[0,1]:
            self.z(qi)
                     
        elif x==[1,0]:
            self.x(qi)
        elif x==[1,1]:
            self.y(qi)
    
    # define square root iswap gates
    def sqriswap(self,q0,q1):
        self.u3(np.pi/2,np.pi/2,-np.pi/2,q0)
        self.u3(2.6974496917414865,-np.pi/2,-np.pi/2,q1)
        self.cx(q0,q1)
        self.u3(3/4*np.pi,-np.pi,-np.pi/2,q0)
        self.u3(1.456908933230463,0.8728233028872143,-1.8815892705543913,q1)
        self.cx(q0,q1)
        self.u3(np.pi/2,np.pi/2,-np.pi,q0)
        self.u3(2.014939288643203,np.pi/2,np.pi/2,q1)


# the circuit of the unitary of interest
nq=10
U = myqc(nq)
for i in range(0,nq,2):
    U.u3(np.pi/2,np.pi/2,-np.pi/2,i)
for i in range(1,nq,2):   
    U.u3(2.6974496917414865,-np.pi/2,-np.pi/2,i)
#U.barrier()
for i in range(0,nq,2):
    U.cx(i,i+1)
U.barrier()

for i in range(0,nq,2):
    U.u3(3/4*np.pi,-np.pi,-np.pi/2,i)
for i in range(1,nq,2):   
    U.u3(1.456908933230463,0.8728233028872143,-1.8815892705543913,i)
#U.barrier()
for i in range(0,nq,2):
    U.cx(i,i+1)
U.barrier()


for i in range(0,nq,2):   
    U.u3(2.014939288643203,np.pi/2,0,i)
for i in range(1,nq,2):
    U.u3(2.6974496917414865,-np.pi/2,-np.pi/2,i)
for i in range(1,nq,2):
    U.cx(i,(i+1)%nq)
U.barrier()

for i in range(0,nq,2):   
    U.u3(1.456908933230463,0.8728233028872143,-1.8815892705543913,i)
for i in range(1,nq,2):
    U.u3(3/4*np.pi,-np.pi,-np.pi/2,i)
for i in range(1,nq,2):
    U.cx(i,(i+1)%nq)
U.barrier()

for i in range(0,nq,2):   
    U.u3(2.014939288643203,np.pi/2,np.pi/2,i)
for i in range(1,nq,2):
    U.u3(np.pi/2,np.pi/2,-np.pi,i)
U.barrier()

# the true phases in this problem
num_p= int(nq/2)+1
true_p= np.sort([np.arccos(np.sin(4*pi/nq*k)**2) for k in range(int(num_p/2))]+ [-1*np.arccos(np.sin(4*pi/nq*k)**2) for k in range(int(num_p/2))])
#true_p= np.sort([true_p_tem[0]]+ list(true_p_tem[1:-1])*2+ [true_p_tem[-1]])
print(true_p)


# the function to generate the power of a unitary
def circ_rep(k,circ):
    num_qubits= circ.num_qubits
    U_rep=QuantumCircuit(num_qubits)
    for i in range(k):
        U_rep=U_rep.compose(U)
    
    return U_rep

# devide a circuit into cycles
def divide_circ(circ):
    circ_list=[[]]
    ind=0
    for ins in circ[0:-1]:
        if 'barrier' == ins[0].name:
            ind=ind+1
            circ_list.append([])
        else:
            circ_list[ind].append(ins)
    return circ_list          

# compute the product of Pauli operators
def prod_pauli(plist):
    pauli_list=np.array([[0,0],[0,1],[1,0],[1,1]])
    s=np.array([0,0])
    for q in plist:
        s=s+pauli_list[q]
    return [i%2 for i in s]

# compute the CNOT gate acting on Pauli operators
def cx_on_pauli(p1,p2):
    
    cx=[[1,0,0,0],[1,1,0,0],[0,0,1,1],[0,0,0,1]]
   
    pauli_2q=[p1[0],p2[0],p1[1],p2[1]]
    new_pauli=np.mod(np.dot(cx,pauli_2q),2)
    new_pauli1=[int(new_pauli[0]),int(new_pauli[2])]
    new_pauli2=[int(new_pauli[1]),int(new_pauli[3])]
    return new_pauli1,new_pauli2

#the product of u3 * p
def u3_pauli(u3,p):   
    new_u3= (u3[0].copy(),u3[1],u3[2])
    theta, phi, lam= new_u3[0].params
    if p==[0,1]:      #p=z
        lam= (lam + np.pi)%(2*np.pi)
    elif p==[1,0]:      #p=x
        theta= (theta+ np.pi)%(2*np.pi)
        lam= (np.pi-lam)%(2*np.pi)
    elif p==[1,1]:                    #p=y
        theta= (theta+ np.pi)%(2*np.pi)
        lam= -lam
    new_u3[0].params=[theta, phi, lam]
    return new_u3

#the product of  p*u3
def pauli_u3(u3,p):    
    new_u3= (u3[0].copy(),u3[1],u3[2])
    theta, phi, lam= new_u3[0].params
    if p==[0,1]:      #p=z
        phi= (phi + np.pi)%(2*np.pi)
    elif p==[1,0]:      #p=x
        theta= (theta+ np.pi)%(2*np.pi)
        phi= (np.pi-phi)%(2*np.pi)
    elif p==[1,1]:                    #p=y
        theta= (theta+ np.pi)%(2*np.pi)
        phi= -phi
    new_u3[0].params=[theta, phi, lam]
    return new_u3

# compute the product of correction Paulis, original single-qubit layer, and random Paulis in randomized compiling
def circ_q1_compile(cor_paulis,g1_list,rand_paulis,nq):
    g1_list_new=[]
    for qb in range(nq):
        new_u3=u3_pauli(g1_list[qb],cor_paulis[qb])
        new_u3= pauli_u3(new_u3,rand_paulis[qb])
        g1_list_new.append(new_u3)   
    
    return g1_list_new

# compile a cycle and return compiled single-qubit layer, two-qubit gates, correction Paulis
def compiled_cycle(bare_cycle,cor_paulis0,nq=10):
    g1_list=[0 for i in range(nq)]
    g2_list=[]
    for ins in bare_cycle:
        if ins[0].num_qubits==1:
            ind=ins[1][0].index
            g1_list[ind]=ins
        else:
            g2_list.append(ins)
    
    pauli_list=[[0,0],[0,1],[1,0],[1,1]] 
    pindex= np.random.randint(4,size=nq)
    rand_paulis= [pauli_list[pindex[i]] for i in range(nq)]
    compiled_ops= circ_q1_compile(cor_paulis0,g1_list,rand_paulis,nq=nq)

    
    cor_paulis1=rand_paulis.copy()
    
   
    for ins in g2_list:
        ind1, ind2= (qubit.index for qubit in ins[1])
        cor_paulis1[ind1],cor_paulis1[ind2]= cx_on_pauli(rand_paulis[ind1],rand_paulis[ind2])
        
    return compiled_ops,g2_list,cor_paulis1

# perform randomizeing compiling on a circuit
def randomized_compiling(bare_circ,nq=10):
    circ1=divide_circ(bare_circ)
    
    qc=myqc(nq)
    cor_paulis0= [[0,0] for i in range(nq)]
    for cycle in circ1:
        compiled_ops,g2_list,cor_paulis1=compiled_cycle(cycle,cor_paulis0,nq)
        
        for i in range(nq):
            qc.append(*compiled_ops[i])
        if g2_list!=[]:
            for ins in g2_list:
                qc.append(*ins)
        
        qc.barrier()
        cor_paulis0= cor_paulis1
    
    for i in range(nq):
        qc.paulis(i,cor_paulis0[i])
    
    return qc

# generate the full circuit for simulation by combining the initial state preparation and final measurement
def full_circ(floq_circ,mode='x',nq=10):
    qc= QuantumCircuit(nq,1)
    qc.h(0)
    qc.barrier()
    
    qc=qc.compose(floq_circ)
    if mode=='y':
        qc.s(0)
    
    qc.h(0)
    qc.barrier()
    qc.measure(0,0)
    return qc

# generate bare circuits with Pauli-X and Pauli-Y measurements for phase estimation 
circ_rep_list=[circ_rep(k,U) for k in range(1,51)]
full_circ_mx_list=list(map(full_circ,circ_rep_list))
full_circ_my_list=list(map(lambda x:full_circ(x,mode='y'),circ_rep_list))

# the noise model for simulation
from qiskit.providers.aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error, coherent_unitary_error,mixed_unitary_error
from scipy.optimize import curve_fit, minimize

def unitary_error_rx(theta):  #exp(-i theta/2 X)
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])

def unitary_error_rxx(theta):          #exp(-i theta/2 X\otimes X)
    return np.cos(theta/2)*np.eye(4)-1j*np.sin(theta/2)*np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])

def unitary_error_rz(theta):  #exp(-i theta/2 Z)
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def unitary_error_rzz(theta):          #exp(-i theta/2 Z\otimes Z)
    return np.cos(theta/2)*np.eye(4)-1j*np.sin(theta/2)*np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

def unitary_error_ry(theta):  #exp(-i theta/2 Y)
    return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

def unitary_error_ryy(theta):          #exp(-i theta/2 Y\otimes Y)
    return np.cos(theta/2)*np.eye(4)-1j*np.sin(theta/2)*np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])

# generate stochastic noise model
def get_noise_model(p=0):
    noise_model=NoiseModel()
    
    
    mu_1q=mixed_unitary_error([(np.array([[1,0],[0,1j]]),p/2),(np.eye(2),1-p/2)])
    mu_2q= mixed_unitary_error([(unitary_error_rzz(np.pi/2),p),(np.eye(4),1-p)])
    
    qerror_1q = mu_1q
    qerror_2q= mu_2q

    noise_model.add_all_qubit_quantum_error(qerror_1q, ['u2','u3','I','X','Y','Z'])
    noise_model.add_all_qubit_quantum_error(qerror_2q, ['cx'])
    
    return noise_model


# the simulation with noise
def noise_sim(circ_list,noise_model,shots=8192):
    result = execute(circ_list, backend=Aer.get_backend('qasm_simulator'),noise_model=noise_model, shots=shots,optimization_level=0).result()
    count_list=[result.get_counts(circ) for circ in circ_list]
    exp_list=[2*count.get('0',0)/shots-1 for count in count_list]
    
    return exp_list

# the simulation with noise by GPU
def noise_sim_gpu(circ_list,noise_model,shots=8192):
    result = execute(circ_list, backend=Aer.get_backend('qasm_simulator'),noise_model=noise_model, 
    shots=shots,optimization_level=0,method='density_matrix_gpu').result()
    count_list=[result.get_counts(circ) for circ in circ_list]
    exp_list=[2*count.get('0',0)/shots-1 for count in count_list]
    
    return exp_list

# the model used to fit data
def model(a,d,nq=10):
    num_p= int(nq/2+1)
    return np.sum([a[i+2*num_p]*(a[i+num_p]**d)*(np.exp(-1j*a[i]*d)) for i in range(num_p)])

# the least-square function
def func(a,x,y):
    est_y= np.array([model(a,d) for d in x])
    dif= np.sum(np.absolute(np.array(y)-est_y)**2)
    return dif

# the starting point for least-square optimization
def generate_x0(phase_est_ft,damping_min=0.8,damping_max=1,coef_min=0.05,coef_max=1,nq=10):
    num_p=int(nq/2+1)
    x0_phase= list(phase_est_ft)
    x0_damping= [np.random.rand()*(damping_max-damping_min)+damping_min for i in range(num_p)]
    x0_coef= [np.random.rand()*(coef_max-coef_min)+coef_min for i in range(num_p)]
    return x0_phase + x0_damping+ x0_coef


# find the optimal estimate of parameters
def find_opt(data_list,depth_list=[i for i in range(1,51)],nq=10,damping_min=0.8,damping_max=1,coef_min=0.05,coef_max=1):
    num_p= int(nq/2+1)
    fourier_amp=list(np.absolute(np.fft.fft(data_list)))
    peaks=find_peaks(fourier_amp)[0]
    peak_vals=[fourier_amp[peak] for peak in peaks]
    loc=[peaks[peak_vals.index(x)] for x in np.sort(peak_vals)[-num_p::]]
    freq=np.fft.fftfreq(50)*2*pi
    phase_est_ft= np.sort(freq[loc])

    phase_bnds=[(min(phase_est_ft)-0.2,max(phase_est_ft)+0.2) for i in range(num_p)]
    damping_bnds= [(damping_min,damping_max) for i in range(num_p)]
    coef_bnds= [(coef_min,coef_max) for i in range(num_p)]
    bnds= phase_bnds+ damping_bnds+ coef_bnds
    
    x0_list=[generate_x0(phase_est_ft,damping_min,damping_max,coef_min,coef_max) for i in range(50)]

    res_list=[]
    for x0 in x0_list:
        res_list.append(minimize(func, x0, (depth_list,data_list),bounds=bnds))

    est_fourier_dis=[np.mean(np.absolute(np.sort(res.x[0:num_p])-phase_est_ft)) for res in res_list]
    min_est =min(est_fourier_dis)
    
    index_filtered= np.where(np.array(est_fourier_dis)-min_est< 0.5*min_est)[0]
    res_list_filtered= [res_list[index] for index in index_filtered]
    funcv_list=[res.fun for res in res_list_filtered]
    min_value= min(funcv_list)
    min_ind= funcv_list.index(min_value)
    return [res.x for res in res_list_filtered][min_ind],min_value

# the phase estimation error
def error(est,nq=10):
    num_p= int(nq/2+1)
    dif=np.sort(est[0:num_p])-np.array(true_p)
    return np.mean(np.absolute(dif))


# the simulation with bare circuits
ts=time()
rep= 10  #the num of repetitions to generate error bars

noise_pro_list= np.linspace(0.001,0.01,11)
error_mean_list=[]
error_std_list=[]

print('noise_pro_list=',noise_pro_list)

for noise_pro in noise_pro_list:
    noise_model= get_noise_model(noise_pro)
    
    error_list=[]
    for i in range(rep):
        xlist=noise_sim_gpu(full_circ_mx_list,noise_model=noise_model,shots=10**5)
        ylist=noise_sim_gpu(full_circ_my_list,noise_model=noise_model,shots=10**5)
        para,val=find_opt(np.array(xlist)+1j*np.array(ylist))
        est_error= error(para)
        error_list.append(est_error)
    error_mean_list.append(np.mean(error_list))
    error_std_list.append(np.std(error_list))

print('bare circuit result:')
print('mean=',error_mean_list)
print('std=',error_std_list)
te=time()
print('time cost:',te-ts)


# the simulation with randomized circuits
ts=time()

rep= 10
shots_tot= 10**5
nrand= 20      # the num of random circuits for each bare circuit
shots= shots_tot/nrand

xdata=[]
ydata=[]
error_list=[]

for k in range(rep):
    cq_list=[]
    for qc in circ_rep_list:
        cq_list.append([randomized_compiling(qc) for i in range(nrand)])

    for noise_pro in noise_pro_list:
        noise_model= get_noise_model(noise_pro)
        
        xlist=[]
        for cqs in cq_list:
            cqs_full=list(map(full_circ,cqs))
            xlist.append(np.mean(noise_sim_gpu(cqs_full,noise_model,shots=shots)))
        xdata.append(xlist)

        ylist=[]
        for cqs in cq_list:
            cqs_full=list(map(lambda x:full_circ(x,mode='y'),cqs))
            ylist.append(np.mean(noise_sim_gpu(cqs_full,noise_model,shots=shots)))
        ydata.append(ylist)

        para,val=find_opt(np.array(xlist)+1j*np.array(ylist))
        error_list.append(error(para))

print('randomized circuit result:')
print('error list=', error_list)

error_chunks= np.transpose(np.split(np.array(error_list),rep))
error_mean_list= [np.mean(chunk) for chunk in error_chunks]
error_std_list= [np.std(chunk) for chunk in error_chunks]
print('mean=',error_mean_list)
print('std=',error_std_list)
te=time()
print('time cost:',te-ts)

print('xdata=',xdata)
print('ydata=',ydata)
