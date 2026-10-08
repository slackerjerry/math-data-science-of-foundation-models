#!/usr/bin/env python3
"""Exact geometry illustrations for Chapter 5, with independent numerical checks.

Run from any directory. Only NumPy and Matplotlib are required.
The default output directory is chapter05/figures, next to this file.
No training data, checkpoints or exercise solutions are used.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, GRAY = '#235d87', '#bb643c', '#30343b', '#a4a9af'
plt.rcParams.update({
    'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
    'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def rotation(t):
    """The chapter's balanced population flow, a(0)=1, b(0)=e1."""
    t = np.asarray(t, dtype=float)
    u = np.stack((1 / np.cosh(t), np.tanh(t)), axis=-1)
    v = np.cosh(t)**2 / (1 + t + 0.5 * np.sinh(2*t))
    a = np.sqrt(v)
    b = a[..., None] * u
    return u, v, a, b


def best_risk(b, task):
    """Refit a scalar head: minimize ||w b - task||^2 under isotropy."""
    b = np.asarray(b, dtype=float)
    w = (b @ task) / np.sum(b*b, axis=-1)
    residual = w[..., None] * b - task
    return np.sum(residual*residual, axis=-1)


def rhs(state):
    a, b1, b2 = state
    return np.array([b2-a*(b1*b1+b2*b2), -a*a*b1, a*(1-a*b2)])


def verify():
    """Compare closed form with RK4 on the original parameter equations."""
    dt = 0.001
    state = np.array([1., 1., 0.])
    max_error = 0.
    for i in range(3000):
        k1 = rhs(state)
        k2 = rhs(state + dt*k1/2)
        k3 = rhs(state + dt*k2/2)
        k4 = rhs(state + dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6
        _, _, a, b = rotation((i+1)*dt)
        max_error = max(max_error, float(np.max(np.abs(state-np.r_[a,b]))))
    assert max_error < 2e-9, max_error
    t = np.linspace(0, 3, 601)
    u, v, a, b = rotation(t)
    r1 = best_risk(b, np.array([1., 0.]))
    r2 = best_risk(b, np.array([0., 1.]))
    unit_error = float(np.max(np.abs(np.sum(u*u, axis=1)-1)))
    risk_error = float(max(np.max(np.abs(r1-np.tanh(t)**2)),
                           np.max(np.abs(r2-1/np.cosh(t)**2))))
    assert unit_error < 1e-14 and risk_error < 1e-14
    assert np.all(np.diff(r1) >= 0) and np.all(np.diff(r2) <= 0)
    assert np.allclose(r1+r2, 1, atol=1e-14)
    maps = np.array([[2.,0.], [1.,1.]])
    target = np.array([1.,0.])
    excess = np.sum((maps-target)**2, axis=1)
    task_risk = best_risk(maps, target)
    assert np.array_equal(excess, [1.,1.])
    assert np.array_equal(task_risk, [0.,0.5])
    # Nonzero scaling changes decoder magnitude, but not the refitted risk.
    assert np.allclose(best_risk(2*b, target), r1, atol=1e-14)
    return dict(rotation_original_ode_rk4_dt=dt,
                rotation_original_ode_max_abs_error=max_error,
                unit_direction_max_error=unit_error,
                optimal_head_projection_max_error=risk_error,
                view_excess=excess.tolist(), best_X1_task_risk=task_risk.tolist(),
                scope='Exact population formulas. RK4 is a sampled numerical check, not a proof.')


def save(fig, out, name):
    fig.savefig(out / (name+'.pdf'), bbox_inches='tight', pad_inches=0.05,
                metadata={'CreationDate': None, 'ModDate': None,
                          'Title': name.replace('_', ' ')})
    fig.savefig(out / (name+'.png'), dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def rotation_figure(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8),
                             gridspec_kw={'width_ratios':[1,1.35]})
    ax = axes[0]
    angle = np.linspace(0, np.pi/2, 301)
    ax.plot(np.cos(angle), np.sin(angle), color=GRAY, lw=1.2)
    times = np.array([0., 0.5, 1., 2.])
    offsets = [(0.07,0.07),(0.07,0.0),(0.07,-0.02),(0.02,0.065)]
    for ti, off in zip(times, offsets):
        u, _, _, _ = rotation(ti)
        ax.annotate('', xy=u, xytext=(0,0),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3))
        ax.plot(*u, 'o', color=BLUE, ms=3)
        ax.text(u[0]+off[0],u[1]+off[1],f'$t={ti:g}$', fontsize=8.5)
    ax.annotate('', xy=(np.cos(.96),np.sin(.96)),
                xytext=(np.cos(.69),np.sin(.69)),
                arrowprops=dict(arrowstyle='->',color=INK,lw=1.1,
                                connectionstyle='arc3,rad=0.13'))
    ax.plot(0,1,'o',mfc='white',mec=INK,ms=4)
    ax.text(-.025,1.09,r'$t\to\infty$',ha='left',fontsize=8.5)
    ax.set(xlim=(-.06,1.34),ylim=(-.12,1.2),aspect='equal',
           xticks=[0,1],yticks=[0,1],xlabel=r'$X_1$ coefficient',
           ylabel=r'$X_2$ coefficient',title=r'(a) Unit direction $u(t)$')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax = axes[1]
    t = np.linspace(0,3,601)
    u, _, _, b = rotation(t)
    r1 = best_risk(b,np.array([1.,0.]))
    r2 = best_risk(b,np.array([0.,1.]))
    ax.plot(t,r2,color=BLUE,lw=2,label=r'$Y=X_2:\ R_2^*=\mathrm{sech}^2t$')
    ax.plot(t,r1,color=ORANGE,lw=2,ls='--',label=r'$Y=X_1:\ R_1^*=\tanh^2t$')
    for task,col in [(np.array([0.,1.]),BLUE),(np.array([1.,0.]),ORANGE)]:
        _,_,_,bm=rotation(times)
        ax.plot(times,best_risk(bm,task),'o',ms=3,color=col)
    ax.set(xlim=(0,3),ylim=(-.025,1.025),xticks=[0,1,2,3],
           yticks=[0,.5,1],xlabel='Gradient-flow time $t$',
           ylabel='Mean squared task risk',title='(b) Refitted optimal heads')
    ax.legend(frameon=False,loc='center right',fontsize=8.5)
    ax.grid(axis='y',color='#e5e6e8',lw=.6)
    fig.tight_layout(w_pad=1.8)
    save(fig,out,'feature_rotation')


def map_figure(out):
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.8))
    target = np.array([1.,0.])
    for ax,f,col,title,risk in zip(axes,[np.array([2.,0.]),np.array([1.,1.])],
            [BLUE,ORANGE],['(a) Decoder magnitude error','(b) Retained direction tilted'],['0',r'1/2']):
        u=f/np.linalg.norm(f)
        line=np.array([-0.15,2.4])[:,None]*u
        ax.plot(line[:,0],line[:,1],color=GRAY,lw=1.2)
        ax.annotate('',xy=f,xytext=(0,0),arrowprops=dict(arrowstyle='->',color=col,lw=2.2))
        projection=np.dot(target,u)*u
        ax.plot(*target,'D',color=INK,ms=4,zorder=4)
        if risk != '0':
            ax.plot([target[0],projection[0]],[target[1],projection[1]],'--',color=INK,lw=1.2)
            ax.plot(*projection,'o',color=INK,ms=3)
            ax.text(0.64,.41,'omitted\ncomponent',fontsize=8)
            ax.text(1.06,1.01,r'$F_{\rm tilt}=(1,1)$',color=col,fontsize=9)
            ax.text(1.22,1.32,'retained span',color='#686d74',fontsize=8)
        else:
            ax.text(1.47,.16,r'$F_{\rm scale}=(2,0)$',color=col,ha='center',fontsize=9)
            ax.text(.12,.50,'retained span',color='#686d74',fontsize=8)
        ax.text(1,-.2,r'$M_*=(1,0)$',ha='center',fontsize=8.5)
        ax.set(xlim=(-.12,2.3),ylim=(-.3,1.55),aspect='equal',
               xticks=[0,1,2],yticks=[0,1],xlabel=r'$X_1$ coefficient',
               ylabel=r'$X_2$ coefficient',title=title)
        ax.text(.5,-.38,r'View excess $=1$'+'\n'+r'Best $Y=X_1$ task risk $='+risk+'$',
                transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.tight_layout(w_pad=2.0)
    save(fig,out,'scale_tilt')


def generate(out):
    out.mkdir(parents=True,exist_ok=True)
    checks=verify()
    (out/'geometry_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (out/'rotation_values.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['time','u1','u2','v','a','b1','b2','optimal_X1_risk','optimal_X2_risk'])
        for t in [0.,.5,1.,2.,3.]:
            u,v,a,b=rotation(t)
            writer.writerow([t,*u,float(v),float(a),*b,float(best_risk(b,np.array([1.,0.]))),float(best_risk(b,np.array([0.,1.])))])
    rotation_figure(out);map_figure(out)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'figures')
    generate(parser.parse_args().out)
