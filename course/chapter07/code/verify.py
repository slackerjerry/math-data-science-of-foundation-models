"""Reproduce declared Chapter 7 calculations. No empirical scaling observations."""
import json
from pathlib import Path
import numpy as np

def direct(s,v):
    p=np.exp(s-np.max(s)); p/=p.sum()
    return p@v,p

def streaming(s,v,blocks):
    state=None
    for ix in blocks:
        if len(ix)==0: continue
        sb,vb=s[ix],v[ix]
        mb=float(sb.max()); weights=np.exp(sb-mb)
        lb=float(weights.sum()); ub=weights@vb
        if state is None: state=(mb,lb,ub); continue
        ma,la,ua=state; m=max(ma,mb)
        state=(m,np.exp(ma-m)*la+np.exp(mb-m)*lb,
               np.exp(ma-m)*ua+np.exp(mb-m)*ub)
    if state is None: raise ValueError('No allowed key')
    m,l,u=state
    return u/l,m+np.log(l)

def main():
    rng=np.random.default_rng(7)
    numerical=[]
    for offset in (0.,1000.):
        s=rng.normal(size=17)+offset; v=rng.normal(size=(17,3)); g=rng.normal(size=3)
        blocks=[np.arange(0,5),np.array([],dtype=int),np.arange(5,13),np.arange(13,17)]
        o,ln=streaming(s,v,blocks); od,p=direct(s,v)
        ds=p*((v-o)@g); dv=p[:,None]*g
        step=1e-4; fd_s=np.zeros_like(s); fd_v=np.zeros_like(v)
        for j in range(len(s)):
            e=np.zeros_like(s);e[j]=step
            fd_s[j]=(g@direct(s+e,v)[0]-g@direct(s-e,v)[0])/(2*step)
            for k in range(v.shape[1]):
                e_v=np.zeros_like(v);e_v[j,k]=step
                fd_v[j,k]=(g@direct(s,v+e_v)[0]-g@direct(s,v-e_v)[0])/(2*step)
        numerical.append(dict(offset=offset,forward=float(np.max(abs(o-od))),
              reconstructed_probability=float(np.max(abs(np.exp(s-ln)-p))),
              score_gradient=float(np.max(abs(ds-fd_s))),value_gradient=float(np.max(abs(dv-fd_v)))))
        assert np.max(abs(ds-fd_s))<1e-7 and np.max(abs(dv-fd_v))<1e-7
    L,d,m,V,T=12,768,3072,32000,1024
    def cost(t): return [L*8*t*d*d,L*4*t*d*m,L*4*t*t*d,2*t*d*V]
    eps=.2
    life=[]
    for M in (0,10,100):
        s=np.sqrt(1+eps*M);N=(1+1/s)/eps;D=(1+s)/eps
        life.append(dict(M=M,N=N,D=D,cost=N*D+M*N,quality=1/N+1/D))
    A=lambda N,D:1+N**(-.5)+D**(-.5)
    B=lambda N,D:1+.8*N**(-.25)+.8*D**(-.75)
    na,da=64.,64.; nb,db=512/3,24.
    out=dict(attention=numerical,flop_parts_T=cost(T),flop_parts_2T=cost(2*T),
        decode=L*(8*d*d+4*d*m+4*(T+1)*d)+2*d*V,cache_bytes=2*L*(T+1)*d*2,
        lifetime=life,scaling=dict(A_at_A=A(na,da),A_at_B=A(nb,db),A_regret=A(nb,db)-A(na,da),
        B_at_B=B(nb,db),B_at_A=B(na,da),B_regret=B(na,da)-B(nb,db),
        separation_64_16=abs(A(64,16)-B(64,16)),separation_16_64=abs(A(16,64)-B(16,64))),
        candidate_utilities=[10*(1-.8**k)-k for k in range(8)])
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
