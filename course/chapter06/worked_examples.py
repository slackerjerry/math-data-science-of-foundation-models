#!/usr/bin/env python3
"""Reproduce Ch6 calculations and figure. Requires numpy, scipy, matplotlib.

All data are generated locally from the stated mathematical models. The finite
lookup experiment is not a trained-model memorization measurement. Independent
checks use enumeration, direct least-squares fits, quadrature, and Monte Carlo.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import platform
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20260930
N_REPEATS = 40000

def close(actual, expected, atol=2e-10, rtol=2e-10):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)

def mc_check(values, expected):
    values = np.asarray(values)
    mean, se = float(values.mean()), float(values.std(ddof=1)/np.sqrt(values.size))
    assert abs(mean-expected) <= 7*se + 1e-10, (mean, expected, se)
    return {'observed': mean, 'theory': float(expected), 'standard_error': se,
            'tolerance': '7 Monte Carlo SE; not a confidence claim about real data'}

def ridge_risk(B, w, sigma2, rho, S):
    n, d = B.shape
    H = B.T@B/n
    G = np.linalg.inv(H+rho*np.eye(d))
    bias = -rho*G@w
    cov = sigma2/n*G@H@G
    return float(bias@S@bias + np.trace(S@cov)), bias, cov

def truncated(theta, a, b, sigma=1):
    # Quadrature remains stable on the explicitly checked finite parameter grid.
    f = lambda z: np.exp(-.5*((z-theta)/sigma)**2)
    mass = quad(f, a, b, epsabs=1e-13, epsrel=1e-13)[0]
    mean = quad(lambda z:z*f(z), a, b, epsabs=1e-13, epsrel=1e-13)[0]/mass
    var = quad(lambda z:(z-mean)**2*f(z), a, b, epsabs=1e-13, epsrel=1e-13)[0]/mass
    return mean, var, mass/(sigma*np.sqrt(2*np.pi))

def main(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    evidence = {'seed':SEED, 'monte_carlo_repeats':N_REPEATS,
                'versions':{'python':platform.python_version(), 'numpy':np.__version__,
                            'scipy':scipy.__version__, 'matplotlib':matplotlib.__version__}}
    # 1. Direct finite-noise enumeration versus general matrix formula.
    B = np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
    w = np.array([1.,-.3]); S = np.array([[1.,.4],[.4,2.]])
    eps = np.array(list(itertools.product([-1.,1.], repeat=4)))
    ridge_tests=[]
    for rho in [0.,.1,1.]:
        fit = np.linalg.solve(B.T@B+4*rho*np.eye(2), B.T@(B@w+eps).T).T
        errors = fit-w
        empirical = np.einsum('ni,ij,nj->n',errors,S,errors).mean()
        theory,_,_ = ridge_risk(B,w,1.,rho,S)
        close(empirical,theory)
        ridge_tests.append({'rho':rho,'enumerated_risk':float(empirical),'formula':theory})
    H=np.diag([1.,.01]); w=np.ones(2); table=[]
    for rho in [0.,.1]:
        G=np.linalg.inv(H+rho*np.eye(2)); bias=-rho*G@w; C=G@H@G/20
        for label,S in [('matched',H),('shifted',np.eye(2))]:
            table.append({'rho':rho,'target':label,'bias2':float(bias@S@bias),
                          'variance':float(np.trace(S@C)), 'risk':float(bias@S@bias+np.trace(S@C))})
    for s in [-1.,-.4,0.,.7,1.]:
        S=np.array([[1.,s],[s,1.]])
        close(bias@S@bias+np.trace(S@C),(111+20*s)/121)
    evidence['ridge']={'finite_noise_cases':ridge_tests,'chapter_table':table}
    # 2. Inverse chi-square moments by quadrature, then independently fit heads.
    inv_moments=[]
    for k in [3,4,5,10,20]:
        value=quad(lambda s:s**(k/2-2)*np.exp(-s/2)/(2**(k/2)*gamma(k/2)),0,np.inf)[0]
        close(value,1/(k-2),atol=2e-9)
        inv_moments.append({'k':k,'integral':value})
    X=rng.normal(size=(N_REPEATS,20,2)); noise=rng.normal(size=(N_REPEATS,20))
    gram=np.einsum('bni,bnj->bij',X,X)
    transfer=[]
    for sin2 in [.02,.10]:
        target=np.sqrt([1-sin2,sin2]); y=X@target+noise
        fit1=np.sum(X[:,:,0]*y,axis=1)/np.sum(X[:,:,0]**2,axis=1)
        risk1=(fit1-target[0])**2+sin2
        fit2=np.linalg.solve(gram,np.einsum('bni,bn->bi',X,y)[...,None])[...,0]
        risk2=np.sum((fit2-target)**2,axis=1)
        transfer.append({'sin2':sin2,'one_feature':mc_check(risk1,sin2+(1+sin2)/18),
                         'two_features':mc_check(risk2,2/17)})
    evidence['transfer']={'inverse_moments':inv_moments,'independent_fits':transfer,'threshold':1/17}
    evidence['validation']={'epsilon_K20_m1000_delta005':float(np.sqrt(np.log(800)/2000)),
                            'zero_error_probability_K1000_m10':float(-np.expm1(1000*np.log1p(-2**-10)))}
    # 3. Exact binomial enumeration, including boundary and single-label cases.
    import math
    teacher_cases=[]
    for n,m,p in [(1,4,.3),(4,7,.3),(20,40,.5),(5,8,0.),(5,8,1.)]:
        mse=0.; blend=0.; alpha=.4
        for k in range(n+1):
            pk=math.comb(n,k)*p**k*(1-p)**(n-k); q=k/n
            for l in range(m+1):
                pl=math.comb(m,l)*q**l*(1-q)**(m-l)
                mse+=pk*pl*(l/m-p)**2
                blend+=pk*pl*((1-alpha)*q+alpha*l/m-p)**2
        theory=p*(1-p)*(1/n+(1-1/n)/m)
        blend_theory=p*(1-p)/n+alpha**2*p*(1-p)*(1-1/n)/m
        close(mse,theory); close(blend,blend_theory)
        teacher_cases.append({'n':n,'m':m,'p':p,'enumerated_mse':mse,'formula':theory})
    evidence['teacher']=teacher_cases
    # 4. Actual stacked least-squares fits for a reused design versus recursions.
    B=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,-1.]])
    w=np.array([.4,-.7]); A=np.linalg.pinv(B); blocks=[]; cumulative=np.zeros(2)
    fit=w.copy(); direct_errors=[]
    for j in range(1,21):
        e=rng.normal(size=4)
        blocks.append(B@fit+e)
        direct=np.linalg.lstsq(np.tile(B,(j,1)),np.concatenate(blocks),rcond=None)[0]
        cumulative+=A@e/j
        close(direct,w+cumulative)
        fit=direct; direct_errors.append(float(np.max(np.abs(direct-w-cumulative))))
    # Exhaust all scalar additive-noise signs for 8 rounds; fit all old rows.
    allsigns=np.array(list(itertools.product([-1.,1.],repeat=8)))
    acc=[]; replace=[]
    for signs in allsigns:
        old=[]; v=0.; u=0.
        for e in signs:
            old.append(v+e); v=float(np.mean(old)); u+=e
        acc.append(v*v); replace.append(u*u)
    close(np.mean(acc),np.sum(1/np.arange(1,9,dtype=float)**2)); close(np.mean(replace),8.)
    evidence['recursion']={'stacked_fit_rounds':20,'maximum_residual':max(direct_errors),
                           'enumerated_noise_histories':256,'accumulated_risk':float(np.mean(acc)),
                           'replacement_risk':float(np.mean(replace))}
    # 5. Truncated-normal moments: quadrature vs normal-CDF formula,
    # finite-difference derivative, and integration-by-parts boundary identity.
    vchecks=[]
    for a,b,sigma in [(-1.,1.,1.),(-1.,3.,1.),(-2.,2.,.7)]:
        worst=0.; boundary_error=0.
        for theta in np.linspace(a,b,17):
            mean,var,accept=truncated(theta,a,b,sigma)
            l,u=(a-theta)/sigma,(b-theta)/sigma
            prob=norm.cdf(u)-norm.cdf(l)
            cdfmean=theta+sigma*(norm.pdf(l)-norm.pdf(u))/prob
            close(mean,cdfmean,atol=1e-9); close(accept,prob,atol=1e-10)
            step=1e-5
            derivative=(truncated(theta+step,a,b,sigma)[0]-truncated(theta-step,a,b,sigma)[0])/(2*step)
            close(derivative,var/sigma**2,atol=3e-9)
            pa=norm.pdf(l)/(sigma*prob); pb=norm.pdf(u)/(sigma*prob)
            boundary=sigma**2*(1-(b-mean)*pb+(a-mean)*pa)
            close(var,boundary,atol=1e-9)
            worst=max(worst,abs(derivative-var/sigma**2)); boundary_error=max(boundary_error,abs(var-boundary))
            assert 0<var<sigma**2
        vchecks.append({'a':a,'b':b,'sigma':sigma,'grid_points':17,
                        'maximum_derivative_error':worst,'maximum_boundary_error':boundary_error})
    biased=truncated(0.,-1.,3.)[0]
    # The soft acceptance formula is checked against direct integration.
    theta,c,sigma,tau=.3,1.,1.,.6
    alpha=tau**2/(sigma**2+tau**2)
    g=lambda z:norm.pdf(z,loc=theta,scale=sigma)*np.exp(-.5*((z-c)/tau)**2)
    mass=quad(g,-np.inf,np.inf)[0]
    mu=quad(lambda z:z*g(z),-np.inf,np.inf)[0]/mass
    vv=quad(lambda z:(z-mu)**2*g(z),-np.inf,np.inf)[0]/mass
    close(mass,np.sqrt(alpha)*np.exp(-(theta-c)**2/(2*(sigma**2+tau**2))))
    close(mu,alpha*theta+(1-alpha)*c); close(vv,alpha*sigma**2)
    proposals=rng.normal(theta,sigma,size=200000)
    mask=rng.random(len(proposals))<np.exp(-.5*((proposals-c)/tau)**2)
    accepted=proposals[mask]
    soft_mc=mc_check(accepted,mu)
    soft_var_mc=mc_check((accepted-mu)**2,vv)
    evidence['verification']={'interval_grid_checks':vchecks,'biased_one_step_mean':biased,
                              'soft_quadrature':{'acceptance':mass,'mean':mu,'variance':vv},
                              'soft_actual_rejection':{'proposals':len(proposals),'accepted':len(accepted),
                                                       'mean_check':soft_mc,'variance_check':soft_var_mc}}
    # 6. Min-norm block fits and all 2^7 sign-noise assignments.
    B=np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
    A=np.array([[1.,2.],[2.,-1.],[1.,-2.]])
    X=np.block([[B,np.zeros((4,3)),np.zeros((4,4))],
                [A,np.eye(3),np.zeros((3,4))]])
    target=np.array([1.,-.5,.2,-.3,.8,2.,1.,-1.,3.])
    noise=np.array(list(itertools.product([-1.,1.],repeat=7)))
    fits=(np.linalg.pinv(X)@(X@target+noise).T).T
    residual=fits@X.T-(X@target+noise)
    close(residual[:,4:],0.)
    assert np.max(np.abs(residual[:,:4]))>.1
    beta=(np.linalg.pinv(B)@(X@target+noise)[:,:4].T).T
    rare=(X@target+noise)[:,4:]-beta@A.T
    close(fits[:,:2],beta); close(fits[:,2:5],rare); close(fits[:,5:],0.)
    composition=[]
    for h,s in [(np.array([1.,0.]),np.array([1.,-1.,0.])),
                (np.array([1.,-1.,0.])@A,np.array([1.,-1.,0.]))]:
        row=np.r_[h,s,np.zeros(4)]
        actual=np.mean(((fits-target)@row)**2)
        theory=(h-s@A)@np.linalg.inv(B.T@B)@(h-s@A)+s@s
        close(actual,theory)
        composition.append({'h':h.tolist(),'s':s.tolist(),'enumerated_risk':float(actual),'formula':float(theory)})
    sign_risks=[]
    for signs in itertools.product([-1.,1.],repeat=2):
        s=np.r_[signs,0.]; row=np.r_[np.zeros(2),s,np.zeros(4)]
        sign_risks.append(float(np.mean(((fits-target)@row)**2)))
    close(np.mean(sign_risks),2+np.sum(A[:2]**2)/4)
    collision=np.array([[1.,1.]])
    close(collision@np.array([1.,-1.]),collision@np.array([0.,0.]))
    evidence['composition']={'noise_assignments':128,'cases':composition,
                             'sign_averaged_risk':float(np.mean(sign_risks)),
                             'rare_residual_max':float(np.max(np.abs(residual[:,4:])))}
    # 7. Explicit, wholly local lookup control with disjoint prefix groups.
    M,r,K=10000,20,16
    suffix=rng.integers(K,size=M); members=rng.choice(M,r,replace=False)
    nonmembers=np.setdiff1d(np.arange(M),members)
    heldout=rng.choice(nonmembers,r,replace=False)
    qtrue=np.full(M,1/K); qtrue[members]=1.
    risk=float(np.mean(1-qtrue)); close(risk,(1-r/M)*(1-1/K))
    threshold=.5*(1+1/K)
    tpr=float(np.mean(qtrue[members]>threshold)); fpr=float(np.mean(qtrue[heldout]>threshold))
    close(tpr,1.); close(fpr,0.)
    greedy=np.zeros(M,dtype=int); greedy[members]=suffix[members]
    evidence['lookup_control']={'M':M,'r':r,'K':K,'population_randomized_risk':risk,
        'membership_TPR':tpr,'membership_FPR':fpr,'supplied_prefix_member_greedy':float(np.mean(greedy[members]==suffix[members])),
        'supplied_prefix_heldout_greedy_observed':float(np.mean(greedy[heldout]==suffix[heldout])),
        'heldout_greedy_expectation_over_worlds':1/K,
        'member_prefix_ids':members.tolist(),'heldout_prefix_ids':heldout.tolist()}
    # Figure: exact finite sums, and a quadrature-computed population map.
    rounds=np.arange(1,21); replacement=rounds.astype(float)
    accumulation=np.cumsum(1/rounds**2)
    steps=np.arange(9); trajectory=[.9]
    for _ in steps[1:]: trajectory.append(truncated(trajectory[-1],-1.,1.)[0])
    with (output/'reuse_verification.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['panel','x','quantity','value'])
        for j,rep,acc in zip(rounds,replacement,accumulation):
            writer.writerow(['reuse',int(j),'replacement_risk_units',float(rep)])
            writer.writerow(['reuse',int(j),'accumulation_risk_units',float(acc)])
        for t,value in zip(steps,trajectory): writer.writerow(['verification',int(t),'population_mean',value])
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.65,2.65),layout='constrained')
    axes[0].plot(rounds,replacement,color='#aa4422',label='Replace')
    axes[0].plot(rounds,accumulation,color='#1d3a54',linestyle='--',label='Accumulate')
    axes[0].set(xlabel='Block count (real block = 1)',ylabel='Risk / initial risk',title='What is retained?',ylim=(0,21))
    axes[0].legend(frameon=False,loc='upper left',fontsize=8)
    axes[1].plot(steps,trajectory,'o-',color='#1d3a54',markersize=3,label='Population fitted mean')
    axes[1].axhline(.4,color='#aa4422',linestyle='--',label='True mean = 0.4')
    axes[1].axhline(0,color='#666666',linestyle=':',label='Verifier center = 0')
    axes[1].set(xlabel='Retraining round',ylabel='Mean',title='Where does verification lead?',ylim=(-.08,1.))
    axes[1].legend(frameon=False,loc='upper right',fontsize=7.5)
    fig.savefig(output/'reuse_verification.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'reuse_verification.png',dpi=180)
    plt.close(fig)
    evidence['figure']={'left':'exact finite sums','right':'deterministic infinite-accepted-batch iteration by quadrature',
                        'right_config':{'sigma':1,'a':-1,'b':1,'theta0':.9,'true_mu':.4},'trajectory':trajectory}
    evidence['status']='all checks passed'
    (output/'checks.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'status':evidence['status'],'output_dir':str(output),'seed':SEED,
                      'finite_enumerations':{'ridge_noise_cases':48,'teacher_parameter_cases':5,
                                             'recursive_noise_histories':256,'composition_noise_histories':128},
                      'interval_grid_points':51,'monte_carlo_repeats':N_REPEATS}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'figures')
    main(parser.parse_args().output_dir.resolve())
