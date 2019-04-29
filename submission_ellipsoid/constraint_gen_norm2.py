from qpsolvers import solve_qp
import numpy as np
from scipy.special import softmax


# Solves inverse cdmp problem with constraint generation
# This is the "heavy" or full version
# Input: list of contexts, list of experts feature expectations, function which computes
#        feature expectations for a specific reward coefficient vector w.
#        for contexts[i], estimation of experts feature expectations are expert_feature_expectations[i]  

def constraint_generation_solver_3(contexts: list, expert_feature_expectations: list,
                                   feature_expectations, iterations = 25, suffix = ""):
    
    DEBUG = True
    assert (len(contexts) == len(expert_feature_expectations))
    num_features = len(expert_feature_expectations[0])
    context_dim = len(contexts[0])
    w_flat_len = num_features * context_dim

    # Initialize "random" policies PI^c(i)
    PI = {}
    init_PI_size = 1
#     W = np.ones((context_dim, num_features))
    W = np.random.uniform(-0.2,1,context_dim*num_features).reshape((context_dim, num_features))
    W = W / np.linalg.norm(W)
    for context in contexts:
        PI[tuple(context)] = [feature_expectations(context @ W).M]

#     for i in range(init_PI_size-1):
#         W = np.random.uniform(-0.2,1,context_dim*num_features).reshape((context_dim, num_features))
#         W = W / np.linalg.norm(W)
#         for context in contexts:
#             PI[tuple(context)].append(feature_expectations(context @ W).M)

    it = 0
    q = np.zeros(w_flat_len)
    P = np.identity(w_flat_len)
    h = -np.ones(init_PI_size*len(contexts))
    G = np.zeros((init_PI_size*len(contexts), w_flat_len))
    # Construct G.
    # c.T @ W @ u  =  (c @ u.T).flatten() @ W.flatten()    (W.flatten() is x)
    i = 0
    for j in range(len(contexts)):
        for feature_expectations_1 in PI[tuple(contexts[j])]:
            G[i] = -np.outer(contexts[j], expert_feature_expectations[j] - feature_expectations_1).flatten()
            i += 1

    # Do some number of times
    while True:
        it += 1
        if DEBUG: print("Iteration ",it,", Num constraints: ",len(G))
        # Minimze (1/2)xPx + qx
        # s.t Gx <= h
        # Solve QP, reconstruct W
        bigW = solve_qp(P, q, G, h).reshape((context_dim, num_features))
        W = bigW / np.linalg.norm(bigW)
        if DEBUG: print("Found W=")
        if DEBUG: print(W)
        if DEBUG: print("With this W:")
        
        #Print info
        for context,exp_feat_exp in zip(contexts,expert_feature_expectations):
            if DEBUG: print("Context: ",context)
            expert_val = context @ W @ exp_feat_exp
            if DEBUG: print("Expert value: ", expert_val)
            my_val = []
            for policy_feat_exp in PI[tuple(context)]:
                my_val.append(context @ W @ policy_feat_exp)
            if DEBUG: print("Value with policies so far: ", my_val)

        # Save matrix
        name = "sol_W/W_" + str(len(contexts)) + "_C_" + str(it) + "_I__V" + str(suffix) +".npy"
        if it % 10 == 0:
            # if DEBUG: print("Writing ",name)
            np.save(name, W)

        # Stop condition TODO: come up with something better
        if it >= iterations:
            return W

        # Update PI^c(i)'s, G, h:        
        # Find most violated constraint
        d = {}
        for context,exp_feat_exp in zip(contexts,expert_feature_expectations):
            expert_val = context @ W @ exp_feat_exp
            my_val = []
            for policy_feat_exp in PI[tuple(context)]:
                my_val.append(context @ W @ policy_feat_exp)
            my_val = np.asarray(my_val).max()
            if not (tuple(context) in d):
                d[tuple(context)] = np.inf
            d[tuple(context)] = min((expert_val-my_val),d[tuple(context)])
            
#         maxim = -np.inf
#         con = contexts[0]
#         for context in contexts:
#             if DEBUG: print(d[tuple(context)])
#             if d[tuple(context)] > maxim:
#                 maxim = d[tuple(context)]
#                 con = context

        pr = 0*np.zeros(len(contexts))
        for i in range(len(contexts)):
            pr[i] = d[tuple(contexts[i])]
        pr = softmax(pr, axis=0)
        if DEBUG: print("probabilities:")
        if DEBUG: print(pr)
        con = contexts[np.random.choice(len(pr), 1, p=pr)[0]]
                
        if DEBUG: print("~most violated context:",con," with value difference ", d[tuple(con)])
        feat_exp = feature_expectations(con @ W).M
        PI[tuple(con)].append(feat_exp)
        h = np.append(h,-np.ones(1))
        G = np.concatenate((G,np.expand_dims(-np.outer(con, exp_feat_exp - feat_exp).flatten(),axis=0)),axis=0)
        try:
            bigW = solve_qp(P, q, G, h).reshape((context_dim, num_features))
        except:
            if DEBUG: print("Failed to add constraint")
            G = G[0:-1]
            h = h[0:-1]

'''
        # Update PI^c(i)'s, G, h:
        h = np.append(h,-np.ones(len(contexts)))
        for context,exp_feat_exp in zip(contexts,expert_feature_expectations):
            feat_exp = feature_expectations(context @ W).M
            PI[tuple(context)].append(feat_exp)
            G = np.concatenate((G,np.expand_dims(-np.outer(context, exp_feat_exp - feat_exp).flatten(),axis=0)),axis=0)

            try:
                bigW = solve_qp(P, q, G, h[:len(G)]).reshape((context_dim, num_features))
            except:
                G = G[0:-1]
                h = h[0:-1]
                
#                 for i in range(len(G)):
#                     Gtag = G.copy()
#                     Gtag = np.delete(Gtag, i, 0)
#                     try:
#                         bigW = solve_qp(P, q, Gtag, h[:len(Gtag)]).reshape((context_dim, num_features))
#                     except:
#                         pass
#                     else:
#                         G = Gtag
#                         h = h[:len(Gtag)]
#                         break
'''