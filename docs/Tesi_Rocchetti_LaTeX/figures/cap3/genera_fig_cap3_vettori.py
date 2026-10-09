import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/sessions/exciting-busy-gauss/mnt/TesiRocchetti')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.conversion import canonical
from src.database import embeddings as E
matplotlib.rcParams.update({'font.size':9,'font.family':'serif','axes.labelsize':9,'axes.titlesize':9.5,
  'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'savefig.dpi':300,'savefig.bbox':'tight'})
C=['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000']
OUT='/sessions/exciting-busy-gauss/mnt/Tesi-Stesura/Tesi_Rocchetti_LaTeX/figures/cap3/'
R='/sessions/exciting-busy-gauss/mnt/TesiRocchetti/DatiElaborati/'

def prep(f,K):
    d=canonical.read(R+f); a=np.asarray(d['angles'][K],float); fps=d['meta']['fps']
    fr=d['cycles']['frames']; sd=d['cycles']['sides']; return d,a,fps,fr,sd
def steps(seg):
    c=E.normalized_cycle(seg); r=np.roll(c,-int(np.argmin(c)))
    v,amp,pk,vl=E.atom(seg); return c,r,v,amp

lab=prep('Ref01_sss_le_shoes04.c3d',('ankle_r','sagittal'))
tel=prep('running_20260220_121922.c3d',('shoulder_r','sagittal'))
# lab: ciclo destro
li=[i for i,s in enumerate(lab[4]) if s=='r'][0]
# tel: primo ciclo destro con atomo valido dopo 5 s
ti=[i for i,(s,(a,b)) in enumerate(zip(tel[4],tel[3])) if s=='r' and a>tel[2]*5 and E.coverage(tel[1][a:b+1])>=E.MIN_COVERAGE and E.atom(tel[1][a:b+1]) is not None][0]

fig,ax=plt.subplots(2,3,figsize=(6.6,4.9),gridspec_kw={'width_ratios':[1.5,1,1]})
rows=[('Laboratorio: Ref01_sss_le_shoes04 (cammino, 200 Hz)',lab,li,'cicli dagli eventi di appoggio del file','caviglia dx, sagittale [°]'),
      ('Telefono: running_20260220_121922 (corsa, 30 Hz)',tel,ti,'cicli da surrogato: flessione della spalla','spalla dx, sagittale [°]')]
for r,(tit,(d,a,fps,fr,sd),i,nota,ylab) in enumerate(rows):
    s,e=fr[i]
    if r==0: w0,w1=0,len(a)
    else:
        w0=max(0,s-int(2*fps)); w1=min(len(a),e+int(2*fps))
    t=np.arange(len(a))/fps
    x=ax[r,0]; x.plot(t[w0:w1],a[w0:w1],color=C[4],lw=1)
    for (cs,ce),ss in zip(fr,sd):
        if ce<w0 or cs>w1: continue
        x.axvline(cs/fps,color='0.7',lw=0.5,zorder=0)
    x.axvspan(s/fps,e/fps,color=C[0],alpha=0.25,lw=0)
    x.set_xlabel('tempo [s]'); x.set_ylabel(ylab)
    x.set_title(tit+'\n'+nota,fontsize=8.5,loc='left')
    c,rr,v,amp=steps(a[s:e+1])
    ph=np.linspace(0,100,len(c))
    y=ax[r,1]; y.plot(ph,c,color='0.6',lw=1,ls='--',label='ricampionato (101 punti)')
    y.plot(ph,rr,color=C[5],lw=1.2,label='ruotato sul minimo')
    y.set_xlabel('ciclo [%]'); y.set_ylabel('[°]')
    z=ax[r,2]; z.stem(np.arange(len(v)),v,linefmt=C[2],markerfmt=' ',basefmt='0.6')
    z.set_ylim(-0.45,0.8); z.set_xlabel('componente'); z.set_ylabel('valore')
    z.text(0.97,0.95,f'32 componenti\nmedia 0, norma 1\nampiezza {amp:.1f}°',transform=z.transAxes,ha='right',va='top',fontsize=7)
for j,lab_ in enumerate(['(a) serie e confini dei cicli','(b) un ciclo normalizzato','(c) vettore della forma']):
    pass
for j,t_ in enumerate(['(a) serie e confini dei cicli','(b) un ciclo normalizzato','(c) vettore della forma']):
    ax[0,j].annotate(t_,xy=(0.5,1.42),xycoords='axes fraction',ha='center',fontsize=9,weight='bold')
h,l=ax[0,1].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.62,-0.01))
fig.tight_layout(rect=(0,0.04,1,0.97),h_pad=1.8)
fig.savefig(OUT+'cap3_vettore_forma.pdf'); fig.savefig(OUT+'cap3_vettore_forma.png')

# Figura b: segnali della sessione telefono, cicli che producono un vettore
d,a,fps,fr,sd=tel; n=len(fr)
it={'trunk':'tronco','spine':'colonna','neck':'collo','shoulder':'spalla','elbow':'gomito','hip':'anca','knee':'ginocchio','ankle':'caviglia'}
pl={'sagittal':'sag.','frontal':'front.','transverse':'trasv.'}
def nome(j,p):
    b,_,s=j.partition('_'); return f"{it[b]}{' '+s.replace('l','sx').replace('r','dx') if s else ''} {pl[p]}"
rows=[]
for (j,p),arr in d['angles'].items():
    arr=np.asarray(arr,float)
    m=sum(1 for s,e in fr if E.coverage(arr[s:e+1])>=E.MIN_COVERAGE and E.atom(arr[s:e+1]) is not None)
    rows.append((nome(j,p),m/n,'presente'))
for k,(why,_) in d['absences'].items():
    if k.count(':')!=2: continue
    _,j,p=k.split(':')
    if p=='sagittal' and j in('hip_r','knee_r','ankle_r'): rows.append((nome(j,p),0,'assente: manca il segmento'))
    if why=='not_observable' and j=='spine' and p=='sagittal': rows.append((nome(j,p),0,'assente: non osservabile per costruzione'))
rows.sort(key=lambda r:(r[1],r[0]))
fig,ax=plt.subplots(figsize=(6.2,4.6))
y=np.arange(len(rows))
col=[C[2] if v>=0.5 else (C[5] if v>0 else '0.75') for _,v,_ in rows]
ax.barh(y,[v*100 for _,v,_ in rows],color=col,height=0.7)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
for yi,(nm,v,st) in zip(y,rows):
    if st!='presente': ax.text(1,yi,st,va='center',fontsize=7,color='0.35')
    else: ax.text(v*100+1,yi,f'{round(v*n)}/{n}',va='center',fontsize=7)
ax.set_xlim(0,112); ax.set_xlabel(f'cicli che producono un vettore della forma [% su {n}]')
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(OUT+'cap3_copertura_telefono.pdf'); fig.savefig(OUT+'cap3_copertura_telefono.png')
print('lab ciclo',lab[3][li],'tel ciclo',tel[3][ti])
