import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/sessions/exciting-busy-gauss/mnt/TesiRocchetti')
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.conversion import canonical, anatomy as A
matplotlib.rcParams.update({'font.family':'serif','font.size':8,'savefig.dpi':300,'savefig.bbox':'tight'})
OUT='/sessions/exciting-busy-gauss/mnt/Tesi-Stesura/Tesi_Rocchetti_LaTeX/figures/'
d=canonical.read('/sessions/exciting-busy-gauss/mnt/TesiRocchetti/DatiElaborati/Ref01_sss_le_shoes04.c3d')
P={k:np.asarray(v,float) for k,v in d['points'].items()}
f=int(np.argmin([np.linalg.norm(P['ankle_l'][i]-P['ankle_r'][i]) for i in range(len(P['head']))]))
p={k:P[k][f] for k in A.POINTS}
lat=p['hip_l']-p['hip_r']; lat/=np.linalg.norm(lat)
up=p['shoulder_c']-(p['hip_l']+p['hip_r'])/2; up-=lat*np.dot(up,lat); up/=np.linalg.norm(up)
fw=np.cross(lat,up)
def fr(q): return np.array([np.dot(q,-lat)*-1*-1, np.dot(q,up)])
def sg(q): return np.array([np.dot(q,fw), np.dot(q,up)])
it={'head':'testa','shoulder_c':'centro spalle','shoulder':'spalla','elbow':'gomito','wrist':'polso','hip':'anca','knee':'ginocchio','ankle':'caviglia','heel':'tallone','toe':'punta'}
def nome(k):
    b,_,s=k.partition('_'); 
    if k=='shoulder_c': return it[k]
    return it[b]+(' '+('sx' if s=='l' else 'dx') if s else '')
segs=[('shoulder_c','head'),('shoulder_l','shoulder_r'),('shoulder_l','elbow_l'),('elbow_l','wrist_l'),('shoulder_r','elbow_r'),('elbow_r','wrist_r'),
 ('hip_l','hip_r'),('hip_l','knee_l'),('knee_l','ankle_l'),('hip_r','knee_r'),('knee_r','ankle_r'),('heel_l','toe_l'),('ankle_l','heel_l'),('ankle_l','toe_l'),('heel_r','toe_r'),('ankle_r','heel_r'),('ankle_r','toe_r')]
fig,axs=plt.subplots(1,2,figsize=(6.4,4.6),gridspec_kw={'width_ratios':[1.25,1]})
mid=(p['hip_l']+p['hip_r'])/2
ax=axs[0]; proj=lambda q:np.array([np.dot(q-mid,lat),np.dot(q-mid,up)])
for a_,b_ in segs:
    A_,B_=proj(p[a_]),proj(p[b_]); ax.plot([A_[0],B_[0]],[A_[1],B_[1]],color='0.45',lw=1.3)
A_,B_=proj(mid),proj(p['shoulder_c']); ax.plot([A_[0],B_[0]],[A_[1],B_[1]],color='0.45',lw=1.3,ls='--')
for k in A.POINTS:
    q=proj(p[k]); c='#D55E00' if k.endswith('_l') else ('#0072B2' if k.endswith('_r') else '#009E73')
    ax.plot(*q,'o',ms=4.5,color=c,zorder=3)
    if k.split('_')[0] in ('heel','toe'): continue
    dx=0.03 if (k.endswith('_l') or not k.endswith('_r')) else -0.03
    lab=nome(k)
    if k.startswith('ankle'): lab+=', tallone, punta'
    ax.text(q[0]+dx,q[1],lab,fontsize=6.5,ha='left' if dx>0 else 'right',va='center')
ax.set_aspect('equal'); ax.axis('off'); ax.set_title('I 18 centri articolari (vista frontale)',fontsize=9)
ax.set_xlim(-0.62,0.62)
ax=axs[1]
side='r'
pr=lambda q:np.array([np.dot(q-p['ankle_'+side],fw),np.dot(q-p['ankle_'+side],up)])
K,An,H,T=[pr(p[x+'_'+side]) for x in ('knee','ankle','heel','toe')]
ax.plot([K[0],An[0]],[K[1],An[1]],color='0.45',lw=1.5)
ax.plot([H[0],T[0]],[H[1],T[1]],color='#009E73',lw=2,label='tallone-punta')
ax.plot([An[0],T[0]],[An[1],T[1]],color='#D55E00',lw=1.6,ls='--',label='caviglia-punta')
ax.plot([An[0],H[0]],[An[1],H[1]],color='0.7',lw=1)
for q,n,dx,dy,ha in ((K,'ginocchio',0.015,0,'left'),(An,'caviglia',0.015,0.02,'left'),(H,'tallone',-0.012,-0.03,'right'),(T,'punta',0.0,-0.035,'center')):
    ax.plot(*q,'o',ms=5,color='#0072B2',zorder=3); ax.text(q[0]+dx,q[1]+dy,n,fontsize=7,ha=ha,va='center')
v1=T-H; v2=T-An
ang=np.degrees(np.arccos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0.0,0.80),fontsize=7,title='asse del piede',title_fontsize=7,alignment='left',handlelength=2.2)
ax.plot([H[0]-0.16],[An[1]],alpha=0)  # spazio a sinistra per la legenda
ax.set_aspect('equal',adjustable='datalim'); ax.axis('off'); ax.set_title('Perché servono i talloni (piede dx, laterale)',fontsize=9)
fig.text(0.33,0.03,'arancione: sinistro   blu: destro   verde: centrali   tratteggio: tronco',ha='center',fontsize=7,color='0.3')
fig.savefig(OUT+'cap3/cap3_vocabolario.pdf'); fig.savefig(OUT+'cap3/cap3_vocabolario.png')
print('frame',f)
