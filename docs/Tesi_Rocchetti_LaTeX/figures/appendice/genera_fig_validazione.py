import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
matplotlib.rcParams.update({'font.family':'serif','font.size':8.5,'savefig.dpi':300,'savefig.bbox':'tight'})
# dati dalla tabella di validazione (appendice), 118 prove
D=[('tronco, frontale',1.000,5.1,5.1,0.0),('tronco, sagittale',0.998,3.8,3.8,0.1),('tronco, trasverso',0.998,11.1,11.1,0.2),
('colonna, trasverso',0.998,14.6,15.7,0.3),('caviglia sx, sagittale',0.996,33.3,33.2,0.8),('caviglia dx, sagittale',0.995,34.9,34.0,0.9),
('anca sx, sagittale',0.996,41.4,32.7,1.2),('anca dx, sagittale',0.997,41.9,34.4,1.2),('ginocchio sx, sagittale',0.986,68.7,69.6,3.2),('ginocchio dx, sagittale',0.988,67.9,67.9,3.2)]
n=[d[0] for d in D]; y=np.arange(len(D))[::-1]
fig,ax=plt.subplots(1,2,figsize=(6.6,3.2),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
ax[0].barh(y+0.18,[d[2] for d in D],height=0.36,color='#56B4E9',label='modello clinico')
ax[0].barh(y-0.18,[d[3] for d in D],height=0.36,color='#E69F00',label='pipeline')
ax[0].set_yticks(y); ax[0].set_yticklabels(n); ax[0].set_xlabel('escursione media [°]'); ax[0].legend(frameon=False,loc='upper right')
ax[1].barh(y,[d[4] for d in D],height=0.55,color=['#D55E00' if d[4]>=3 else '#009E73' for d in D])
for yi,d in zip(y,D): ax[1].text(d[4]+0.08,yi,f'{d[4]:.1f}°  (r={d[1]:.3f})',va='center',fontsize=7)
ax[1].set_xlim(0,5.2); ax[1].set_xlabel('scarto quadratico medio [°]')
for a in ax: a.spines[['top','right']].set_visible(False)
fig.tight_layout()
O='/sessions/exciting-busy-gauss/mnt/Tesi-Stesura/Tesi_Rocchetti_LaTeX/figures/appendice/'
fig.savefig(O+'app_validazione.pdf'); fig.savefig(O+'app_validazione.png')
