import numpy as np
import ot
import random
from scipy.spatial.distance import cdist


def k_barycenter(Q, k, lambd):
    c = len(Q)  
    m = len(Q[0])  

    H = np.zeros((c,k)) 
    for i in range(k):
        point = random.randint(0,m-1)
        for j in range(c):
            H[j,i] = Q[j,point]
    
    t = 1.1
    eta = 0.5
    b = np.ones(m)/m  

    a1 = np.ones(k)/k
    a2 = np.ones(k)/k
    
    a1_former = np.zeros(k)
    H_former = np.zeros((c,k))
    
    convergence_a1=pow(10,-2)
    convergence_H=pow(10,-2)

    while np.linalg.norm(H-H_former)>convergence_H:
        MHQ = cdist(H.transpose(), Q.transpose(), metric='euclidean')
        a1 = np.ones(k)/k
        a2 = np.ones(k)/k
        a1_former = np.zeros(k)
        while np.linalg.norm(a1-a1_former)>convergence_a1:
            beta = (t+1)/2
            a = (1-1/beta)*a1 + (1/beta)*a2
            result, dual = ot.sinkhorn(a, b, MHQ, lambd, verbose=False, log=True)
            alpha = dual['u']
            alpha = (-beta)*alpha
            alpha = np.exp(alpha)
            a2 = a2
            a2_n = a2*alpha
            
            if np.sum(np.isinf(a2_n))==1:
                a2 = np.zeros((len(a2),))
                a2[np.isinf(a2_n)]=1
            elif np.all(a2_n==0):
                a2 = np.ones((len(a2),))/len(a2)
            else:
                a2 = a2_n/np.sum(a2_n)
            a1_former = a1
            a1 = (1-1/beta)*a1 + (1/beta)*a2
            t+=1       

        a = a1
        T = ot.sinkhorn(a, b, MHQ, lambd, verbose=False)
        T = T.transpose()
        diag_a_reverse = np.diag(1/a)
        H_former = H
        H = (1-eta)*H + eta*np.dot(np.dot(Q,T),diag_a_reverse)       
    
    return H 


def label_reg(a, b, cost_matrix, label, classes, lambd, p=0.5, epsilon=1e-6, iters=5, eta=1):
    G = np.zeros((len(cost_matrix),len(cost_matrix[0])))
    C = np.zeros((len(cost_matrix),len(cost_matrix[0])))    
    label_matrix = np.zeros((classes,len(label)))

    for i in range(len(label)):
        label_matrix[label[i],i] = 1 
        

    for step in range(iters):
        C = G + cost_matrix       
        result = ot.sinkhorn(a, b, C, lambd, verbose=False)        
        G = np.dot(label_matrix, result)
        G = np.dot(label_matrix.transpose(), G)
        G = p*np.power(G+epsilon,p-1)*eta


    return result  

def label_propagation_analysis(opt_result, label, classes):
    
    label_matrix = np.zeros((classes,len(label)))
    for i in range(len(label)):
        label_matrix[label[i],i] = 1
    
    result = np.dot(label_matrix, opt_result)

    return np.max(result,axis=0), np.argmax(result,axis=0)


def label_propagation_analysis_class_change(opt_result, label, classes, train_label):
    import numpy as np

    label_matrix = np.zeros((classes, len(label)))
    for i in range(len(label)):
        label_matrix[label[i], i] = 1

    result = np.dot(label_matrix, opt_result)
    predicted_classes = np.argmax(result, axis=0)

    threshold = int(len(train_label) / classes * 0.2)

    for cls in range(classes):
        cls_indices = np.where(predicted_classes == cls)[0]
        correct_indices = [idx for idx in cls_indices if train_label[idx] == cls]
        num_correct_samples = len(correct_indices)
        if num_correct_samples < threshold:
            additional_samples = np.where(train_label == cls)[0]
            additional_samples = [idx for idx in additional_samples if idx not in correct_indices]
            additional_samples = np.array(additional_samples, dtype=int)  
            cls_probs = opt_result[cls, additional_samples]

            sorted_indices = np.argsort(cls_probs)[::-1]
            sorted_indices = np.array(sorted_indices, dtype=int)

            num_to_add = min(len(additional_samples), int(threshold - num_correct_samples))

            if num_to_add > 0 and len(sorted_indices) > 0:
                for idx in additional_samples[sorted_indices][:num_to_add]:
                    result[:, idx] = 0
                    result[cls, idx] = 1

    return np.max(result, axis=0), np.argmax(result, axis=0)
