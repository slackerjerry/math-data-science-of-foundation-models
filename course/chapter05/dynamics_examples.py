"""Standard-library numerical checks for the Chapter 5 dynamics."""
import json,math
from fractions import Fraction as Q

def rk4(f,x,dt,steps):
    x=list(x)
    for _ in range(steps):
        k1=f(x)
        k2=f([a+dt*b/2 for a,b in zip(x,k1)])
        k3=f([a+dt*b/2 for a,b in zip(x,k2)])
        k4=f([a+dt*b for a,b in zip(x,k3)])
        x=[a+dt*(b+2*c+2*d+e)/6 for a,b,c,d,e in zip(x,k1,k2,k3,k4)]
    return x

def check():
    errors=[]
    for s in [0.7,1.,2.]:
        for a,b in [(0.1,0.1),(.1,-.1),(0.,1.),(1.5,.2),(0.,0.)]:
            f=lambda x: [(s-x[0]*x[1])*x[1],(s-x[0]*x[1])*x[0]]
            af,bf=rk4(f,[a,b],.001,2000)
            err=abs(af*af-bf*bf-(a*a-b*b))
            errors.append(err);assert err<1e-9,err
            if a==b and a!=0:
                exact=s/(1+(s/(a*b)-1)*math.exp(-4*s))
                assert abs(af*bf-exact)<1e-10
            if a==-b and a!=0:
                v0=-a*b
                exact=-s*v0*math.exp(-4*s)/(s+v0*(1-math.exp(-4*s)))
                assert abs(af*bf-exact)<1e-10
    # Exact rational discrete invariant and overshoot example.
    a,b,s,eta=Q(5),Q(1,20),Q(1),Q(1)
    d=s-a*b
    ap,bp=a+eta*d*b,b+eta*d*a
    assert ap*ap-bp*bp==(1-eta*eta*d*d)*(a*a-b*b)
    assert ap*bp==Q(7657,400)
    assert (ap*bp-1)**2/2>162
    f=lambda x:[(1-x[0]-2*x[0]**3)/2]
    theta=rk4(f,[0.],1e-5,100)[0]
    small_t_ratio=theta**2/.001**2
    assert abs(small_t_ratio-.25)<.0002
    for theta in [-2.,0.,.2,.8,2.]:
        r=[theta-1,theta**2]
        j=[1.,2*theta]
        g=sum(a*b for a,b in zip(r,j))/2
        direct=sum(a*b for a,b in zip(r,j))*(-g)
        identity=-sum(a*b for a,b in zip(r,j))**2/2
        assert abs(direct-identity)<1e-10
    # The Chapter 3 vocabulary update, evaluated without automatic differentiation.
    aa=1/(2*math.sqrt(2));bb=aa+1/6;cc=bb/math.sqrt(1+bb*bb)
    p=1/(1+math.exp(-2*cc));eta=.1
    old=-math.log(p)
    new=-math.log(1/(1+math.exp(-(2*cc+4*eta*cc*cc*(1-p)))))
    assert new<old
    result=dict(max_flow_invariant_error=max(errors),small_time_second_prediction_ratio=small_t_ratio,
                overshoot_product=str(ap*bp),token_probability=p,token_loss_before=old,token_loss_after=new)
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':check()

