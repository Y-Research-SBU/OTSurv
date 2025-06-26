"""
modified from https://github.com/rhfeiyang/PPOT
"""
import torch

class SK_Class(torch.nn.Module):
    def set_rho(self,rho):  # Used to control the sparsity of pseudo labels
        if hasattr(self,"rho"):
            self.rho=rho
    def get_rho(self):
        if hasattr(self,"rho"):
            return self.rho
        else:
            return None
    def set_gamma(self,gamma):
        if hasattr(self,"gamma"):
            self.gamma=gamma
    def get_gamma(self):
        if hasattr(self,"gamma"):
            return self.gamma
        else:
            return None

class SemiCurrSinkhornKnopp(SK_Class):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-6, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon  # Regularization parameter, controls the smoothness of the Sinkhorn-Knopp algorithm.
        self.gamma = gamma  # Regularization coefficient, used to control the strength of semi-relaxed constraints.
        self.stoperr = stoperr  # Error threshold for stopping iteration.
        self.numItermax = numItermax
        self.rho = rho  # Total mass constraint, used to control the allocation quality of pseudo labels.
        self.b = None
        self.semi_use = semi_use  # Whether to enable semi-relaxed constraints.
        self.prior = prior.reshape(-1,1) if prior is not None else None  # Prior distribution, used to adjust the allocation of pseudo labels to make them more consistent with the target distribution.
        # print(f"prior: {prior}")
        # print(f"semi_use: {semi_use}")
        # print(f"epsilon: {epsilon}")
        # print(f"sk_numItermax: {numItermax}")
    
    # cost_forward() function: Calculate pseudo label allocation matrix (transport plan matrix).
    # Steps:
    # Initialize Pa and Pb, representing the initial distribution of samples and prior distribution of classes respectively.
    # Use Sinkhorn-Knopp algorithm for iterative updates, calculate the final transport plan matrix plan through successive scaling of a and b.
    # If semi_use parameter is set, adjust the update of b to ensure satisfaction of semi-relaxed constraints.
    # Return the final pseudo label allocation matrix plan.
    # @torch.no_grad()
    def cost_forward(self, cost, final=True, count=False, pred_order=None):
        device = cost.device
        cost=cost.double() # Cost matrix cost
        n=cost.shape[0]  # n is the number of samples
        k=cost.shape[1]  # k is the number of classes
        mu = torch.zeros(n, 1).to(device)
        expand_cost = torch.cat([cost, mu], dim=1)  # Add a column of zeros at the end of the cost matrix to support semi-relaxed constraints.
        Q = torch.exp(- expand_cost / self.epsilon)  # Initial transport plan matrix Q

        # prior distribution
        # Pa represents the initial distribution of samples, which is uniform
        Pa = torch.ones(n, 1).to(device) / n  # how many samples 。
        # Pb represents the prior distribution of classes, by uniformly distributing the number of classes to the total mass constraint rho
        Pb = self.rho * torch.ones(Q.shape[1], 1).to(device) / k # how many prototypes
        # If prior exists, adjust Pb accordingly
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=0)
                pred_order = pred_distribution.argsort(descending=True)
            # print(f"pred_order: {pred_sort_order}")
            Pb[pred_order,:] = self.prior * self.rho
        Pb[-1] = 1 - self.rho

        # a and b are vectors used for progressive scaling, adjusted iteratively to satisfy constraints.
        # init b
        b = torch.ones(Q.shape[1], 1).double().to(device) / Q.shape[1] if self.b is None else self.b

        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)

            # If semi_use is set, perform power operation adjustment on b to ensure satisfaction of semi-relaxed constraints.
            if self.semi_use:
                b[:-1,:] = torch.pow(b[:-1,:], fi)

            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = a*Q*b.T
        # If final=True, multiply plan by the number of samples to ensure satisfaction of total mass constraint.
        if final:
            plan*=Q.shape[0]
        self.b=b # for two view speed up
        # print(f"sk_iter: {iternum}"
        # print(iternum,end=" ")
        # scale the plan
        # plan = plan / torch.sum(plan, dim=1, keepdim=True)
        
        # Reason for removing the last column.
        # Semi-relaxed constraint: In semi-relaxed optimal transport problems, the last column is to introduce a virtual class to absorb some remaining mass of samples, thus achieving semi-relaxed effect
        plan = plan[:, :-1].float()  
        # loss = (plan * cost).sum()
        # print(f"sk loss: {loss}")

        # plan is the final pseudo label allocation matrix, i.e., the transport plan matrix from samples to classes.
        return (plan, iternum) if count else plan

    # forward(self, logits) method
    # Function: Calculate negative log-likelihood based on model logits and call cost_forward to calculate final pseudo label allocation.
    # Steps:
    # Convert logits to probability distribution and calculate its negative logarithm.
    # Calculate pseudo label allocation through cost_forward method.
    # @torch.no_grad()
    def forward(self, logits):
        # logits are the output scores of the model for samples, representing the scores of samples belonging to each class.
        logits = logits.detach()
        logits = -torch.log(torch.softmax(logits, dim=1))  # Get cost matrix
        return self.cost_forward(logits)  # Use the calculated cost matrix to call cost_forward, generating pseudo label allocation matrix.


# This is the stable version of SemiCurrSinkhornKnopp class, improving numerical stability.
# Main difference: Added u and v variables for stable computation during iteration.
class SemiCurrSinkhornKnopp_stable(SemiCurrSinkhornKnopp):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-10, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__(num_iters, epsilon, gamma, stoperr, numItermax, rho, semi_use, prior)
        self.u = None  # u and v: dual variables, used for numerical adjustment during iteration to avoid numerical overflow caused by excessive scaling operations.
        self.v = None
    
    # reset(self) method: Used to reset u, v and b.
    # Before each iteration, if the shape of dual variables is inconsistent with current input data, call this method to reinitialize variables, ensuring correctness of computation process.
    def reset(self):
        self.u=None
        self.v=None
        self.b=None
    
    # Calculate pseudo label allocation matrix (transport plan matrix)
    # @torch.no_grad()
    def cost_forward(self, cost, final=True,count=False, pred_order=None):
        # Process input cost matrix cost:
        # Convert cost to double precision type and add a column of zeros at the end to form expand_cost for supporting semi-relaxed optimal transport.
        # If dual variables u and v don't exist or shapes don't match, call reset() method to reinitialize.
        device = cost.device
        cost=cost.double()
        n=cost.shape[0]
        k=cost.shape[1]
        if self.u is not None and self.u.shape[0]!=n:
            self.reset()
        mu = torch.zeros(n, 1).to(device)
        expand_cost = torch.cat([cost, mu], dim=1)

        # Initialize dual variables u and v and initial plan matrix Q
        if self.u is None:
            u = torch.zeros(n, 1).to(device)
            v= torch.zeros(k+1, 1).to(device)
            Q = torch.exp(- expand_cost / self.epsilon)
        else:
            u=self.u
            v=self.v
            Q = torch.exp((u - expand_cost + v.T) / self.epsilon)

        # Pa and Pb represent the initial distribution of samples and prior distribution of classes respectively
        # prior distribution
        Pa = torch.ones(n, 1).to(device) / n  # how many samples
        Pb = self.rho * torch.ones(Q.shape[1], 1).to(device) / k # how many prototypes
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=0)
                pred_order = pred_distribution.argsort(descending=True)
            # print(f"pred_order: {pred_sort_order}")
            Pb[pred_order,:] = self.prior * self.rho
        Pb[-1] = 1 - self.rho
        fi = self.gamma / (self.gamma + self.epsilon)

        # Initialize scaling vector b and weight vector w
        b = torch.ones(Q.shape[1], 1, dtype=Q.dtype).to(device) / Q.shape[1] if self.b is None else self.b
        w = torch.exp(v[:-1, :] * (fi - 1) / self.epsilon)
        
        err = 1
        last_b = b.clone()
        iternum = 0
        stabled=False
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)
            if self.semi_use:
                # b[:-1,:] = torch.pow(b[:-1,:], fi) * w
                b = torch.cat([
                    torch.pow(b[:-1, :], fi) * w,
                    b[-1:, :]
                ], dim=0)

            # print((a*Q*b.T).sum(), err)

            err = torch.norm(b - last_b)
            # In each iteration, if the maximum values of a and b exceed a certain threshold, enable update of dual variables u and v
            if max(a.max(), b.max())>1e8:
                # print(f"stabled at {iternum}")
                # u += self.epsilon * torch.log(a)
                # v += self.epsilon * torch.log(b + torch.finfo(b.dtype).eps)
                # w *= torch.pow(b[:-1,:], fi-1)
                u = u + self.epsilon * torch.log(a)
                v = v + self.epsilon * torch.log(b + torch.finfo(b.dtype).eps)
                w = w * torch.pow(b[:-1,:], fi-1)
                Q = torch.exp((u - expand_cost + v.T) / self.epsilon)
                b = torch.ones_like(b)  # Replace in-place operation
                # b[:,:] = 1
                # a[:,:] = 1
                stabled = True
            else:
                stabled=False

            last_b = b.clone()
            iternum += 1

        plan = Q if stabled else a*Q*b.T
        if final:  # If final=True, multiply plan by the number of samples to satisfy total mass constraint.
            plan*=Q.shape[0]
        self.b=b # for two view speed up
        self.u=u
        self.v=v
        # print(f"sk_iter: {iternum}")
        # print(iternum,end=" ")

        plan = plan[:, :-1].float()
        # loss = (plan * cost).sum()
        # print(f"sk_stable loss: {loss}")
        # count is a boolean variable indicating whether to return iteration count
        return (plan, iternum) if count else plan

