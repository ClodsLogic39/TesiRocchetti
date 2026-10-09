import sys; sys.dont_write_bytecode=True
sys.path.insert(0,'/sessions/exciting-busy-gauss/mnt/TesiRocchetti')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.conversion import canonical, segmentation as S
matplotlib.rcParams.update({'font.size':9,'font.family':'serif','savefig.dpi':300,'savefig.bbox':'tight'})
C=['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000']
OUT='/sessions/exciting-busy-gauss/mnt/Tesi-Stesura/Tesi_Rocchetti_LaTeX/figures/cap3/'
R='/sessions/exciting-busy-gauss/mnt/TesiRocchetti/DatiElaborati/'
nomi={'source_events':'eventi di appoggio del file','toe_coordinate':'traiettoria della punta del piede',
 'toe_coordinate_step':'passo della punta del piede','hip_flexion':"flessione dell'anca",'shoulder_flexion':'flessione della spalla'}
fig,axs=plt.subplots(1,2,figsize=(6.8,2.7),sharey=True)
for ax,(f,tit,sport) in zip(axs,[('Ref01_sss_le_shoes04.c3d','Laboratorio (cammino)','walking'),('running_20260220_121922.c3d','Telefono (corsa)','running')]):
    d=canonical.read(R+f); ctx={'landmarks':d['points'],'joints':S._joints_by_plane(d['angles'])}
    scelto=d['cycles']['method']; order=S.STRATEGIES[sport]
    vals,cols,labs=[],[],[]
    passato=False
    for m in order:
        if m=='source_events':
            v=1.0 if scelto=='source_events' else 0.0
        else: v=S.INPUT_COVERAGE[m](ctx)
        if passato: st='non necessario'; c='0.85'
        elif m==scelto: st='scelto'; c=C[2]; passato=True
        elif m=='source_events': st='assenti nel formato'; c='0.6'
        else: st='scartato' if v<S.MIN_INPUT_COVERAGE else 'nessun ciclo'; c=C[5]
        vals.append(v*100); cols.append(c); labs.append(st)
    y=np.arange(len(order))[::-1]
    ax.barh(y,vals,color=cols,height=0.6)
    for yi,v,st in zip(y,vals,labs):
        ax.text(min(v,100)+2 if v<70 else v-2,yi,f'{st}' + ('' if st in('non necessario','assenti nel formato') else f' ({v:.0f}%)'),
                va='center',ha='left' if v<70 else 'right',fontsize=7.5,color='white' if (v>=70 and st=='scelto') else '0.15')
    ax.axvline(S.MIN_INPUT_COVERAGE*100,color='0.3',ls='--',lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([nomi[m] for m in order])
    ax.set_xlim(0,100); ax.set_xlabel('copertura del segnale [%]'); ax.set_title(tit,fontsize=9)
    ax.spines[['top','right']].set_visible(False)
axs[1].text(S.MIN_INPUT_COVERAGE*100+1,-0.85,'soglia 50%',fontsize=7,color='0.3')
fig.tight_layout()
fig.savefig(OUT+'cap3_ripiego_cicli.pdf'); fig.savefig(OUT+'cap3_ripiego_cicli.png')
