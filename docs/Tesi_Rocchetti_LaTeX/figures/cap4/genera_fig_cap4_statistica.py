import json, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
matplotlib.rcParams.update({'font.family':'serif','font.size':8.5,'savefig.dpi':300,'savefig.bbox':'tight'})
O=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(O,'cap4_statistica_dati.json')))
C=['piano','rampa75','rampa10']; NOMI={'piano':'piano','rampa75':'rampa 7.5%','rampa10':'rampa 10%'}
COL={'piano':'#009E73','rampa75':'#E69F00','rampa10':'#D55E00'}
N=D['sessioni']
# pannello sinistro: alert per sessione, per gravita'
SEV=[('critical','#B2182B'),('warning','#EF8A62'),('info','#67A9CF'),('data_quality','#BBBBBB')]
fig,ax=plt.subplots(1,2,figsize=(6.8,3.3),gridspec_kw={'width_ratios':[0.75,1.6]})
x=np.arange(len(C)); base=np.zeros(len(C))
for s,col in SEV:
    v=np.array([D['severita'][c].get(s,0)/N[c] for c in C])
    if v.sum()==0: continue
    ax[0].bar(x,v,bottom=base,color=col,width=0.6,label=s); base+=v
for xi,c in zip(x,C): ax[0].text(xi,base[xi]+0.1,f'{base[xi]:.2f}',ha='center',fontsize=7.5)
ETI={'piano':'piano','rampa75':'rampa\n7.5%','rampa10':'rampa\n10%'}
ax[0].set_xticks(x); ax[0].set_xticklabels([ETI[c] for c in C],fontsize=7.5); ax[0].set_ylim(0,8.2)
ax[0].set_ylabel('alert per sessione'); ax[0].legend(frameon=False,fontsize=6.5,loc='upper left',ncol=2,columnspacing=0.8,handlelength=1.2)
# pannello destro: tipi piu' frequenti, alert per sessione
TIPI=[('knee|sagittal|peak_min|warning','ginocchio, iperestensione\n(oltre l\'intervallo del cammino)'),
      ('knee|sagittal|peak_min|critical','ginocchio, iperestensione\n(oltre il limite anatomico)'),
      ('hip|sagittal|peak_max|warning','anca, flessione massima'),
      ('hip|sagittal|rom|warning','anca, escursione sagittale'),
      ('hip|sagittal|peak_min|warning','anca, estensione insufficiente'),
      ('knee|sagittal|rom|warning','ginocchio, escursione sagittale'),
      ('hip|frontal|rom|info','anca, escursione frontale')]
y=np.arange(len(TIPI))[::-1]; h=0.26
for k,c in enumerate(C):
    v=[D['tipi'][c].get(t,0)/N[c] for t,_ in TIPI]
    ax[1].barh(y+(1-k)*h,v,height=h,color=COL[c],label=NOMI[c])
ax[1].set_yticks(y); ax[1].set_yticklabels([n for _,n in TIPI],fontsize=7.5)
ax[1].set_xlabel('alert per sessione'); ax[1].legend(frameon=False,fontsize=7,loc='lower right')
for a in ax: a.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(O,'cap4_statistica.pdf')); fig.savefig(os.path.join(O,'cap4_statistica.png'))
