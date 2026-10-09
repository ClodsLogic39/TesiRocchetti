import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
matplotlib.rcParams.update({'font.family':'serif','font.size':8.5,'savefig.dpi':300,'savefig.bbox':'tight'})
C={'or':'#E69F00','az':'#56B4E9','ve':'#009E73','bl':'#0072B2','ro':'#D55E00','ro2':'#CC79A7','gr':'0.85'}
OUT='/sessions/exciting-busy-gauss/mnt/Tesi-Stesura/Tesi_Rocchetti_LaTeX/figures/'
def box(ax,x,y,w,h,title,body='',fc='white',ec='0.2',tsize=9,bsize=7.5,lw=1):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.015',fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h-0.022,title,ha='center',va='top',fontsize=tsize,weight='bold')
    if body: ax.text(x+w/2,y+h-0.075,body,ha='center',va='top',fontsize=bsize,linespacing=1.35)
def arr(ax,a,b,txt='',style='-|>',ls='-',col='0.25',off=(0,0.018),rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,color=col,lw=1,ls=ls,connectionstyle=f'arc3,rad={rad}'))
    if txt: ax.text((a[0]+b[0])/2+off[0],(a[1]+b[1])/2+off[1],txt,ha='center',va='bottom',fontsize=7,color='0.25')
def canvas(w,h):
    fig=plt.figure(figsize=(w,h)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(-0.015,1.015); ax.set_ylim(-0.02,1.02); ax.axis('off'); return fig,ax

# ---------- F1 pipeline ----------
fig,ax=canvas(6.8,4.3)
# cinque colonne larghe W con intervalli di 0.0625, abbastanza per frecce ed etichette
W=0.15; GAP=0.0625; X=[i*(W+GAP) for i in range(5)]; g=0.012
def col(i): return X[i]
ym,hm=0.40,0.32; yb,hb=0.02,0.28; bs=7
box(ax,col(0),ym,W,hm,'Sorgenti','C3D di laboratorio\n(marker, eventi,\nmodello clinico)\n\npunti da video\n(MediaPipe)',fc='#f3f3f3',bsize=bs)
box(ax,col(1),ym,W,hm,'Conversione','etichette, 18 centri\narticolari, terne,\nangoli, cicli,\nassenze',fc='#fff3dc',bsize=bs)
box(ax,col(2),ym,W,hm,'C3D canonico','POINT, ANALOG\ne gruppo BIOMECH:\nprovenienza,\ncopertura, assenze,\ncicli, versione',fc='#fde2cf',lw=1.6,bsize=bs)
box(ax,col(3),ym,W,hm,'Caricamento','righe del database,\nvettori della forma,\nvalutazione\ndegli alert',fc='#fff3dc',bsize=bs)
box(ax,col(4),ym,W,hm,'Database','PostgreSQL\ncon pgvector:\nsessioni, angoli,\ncicli, vettori,\nscostamenti',fc='#dcecf8',bsize=bs)
box(ax,col(3),0.80,W,0.18,'Riferimenti','file dichiarativo\ncon le fonti',fc='#f3f3f3',bsize=bs)
box(ax,col(2),yb,W,hb,'Visualizzatore','manichino 3D,\ngiunto rosso nei\nfotogrammi dello\nscostamento',fc='#f7e3ef',bsize=bs)
box(ax,col(3),yb,W,hb,'Server MCP','20 strumenti,\nlettura del DB,\nrisposte con\ncopertura e avvisi',fc='#dcf1e8',bsize=bs)
box(ax,col(4),yb,W,hb,'Agente','qualunque\nclient MCP',fc='#f3f3f3',bsize=bs)
yc=ym+hm/2
for i in range(4): arr(ax,(col(i)+W+g,yc),(col(i+1)-g,yc))
c3=col(3)+W/2
arr(ax,(c3,0.80-g),(c3,ym+hm+g)); ax.text(c3+0.01,0.755,'intervalli',fontsize=7,color='0.25',va='center')
c2=col(2)+W/2
arr(ax,(c2,ym-g),(c2,yb+hb+g)); ax.text(c2-0.01,0.36,'legge il file',fontsize=7,color='0.25',ha='right',va='center')
a,b=(col(4)+0.04,ym-g),(col(3)+W-0.02,yb+hb+g); arr(ax,a,b)
ax.text((a[0]+b[0])/2+0.04,(a[1]+b[1])/2+0.015,'lettura',fontsize=7,color='0.25',ha='left',va='center')
arr(ax,(col(3)+0.035,yb+hb+g),(col(3)+0.035,ym-g),ls=':',col=C['ro']); ax.text(col(3)+0.045,0.35,'avvia\ndalla chat',fontsize=7,color=C['ro'],ha='left',va='center')
yv=yb+hb/2
arr(ax,(col(3)-g,yv),(col(2)+W+g,yv)); ax.text((col(3)+col(2)+W)/2,yv+0.012,'apre',fontsize=7,color='0.25',ha='center',va='bottom')
arr(ax,(col(3)+W+g,yv+0.03),(col(4)-g,yv+0.03)); arr(ax,(col(4)-g,yv-0.03),(col(3)+W+g,yv-0.03))
xb=(col(1)+W+col(2))/2
ax.plot([xb,xb],[ym-0.02,ym+hm+0.04],ls='--',color=C['ro'],lw=0.9)
ax.text(xb,ym+hm+0.05,'confine: il file canonico',fontsize=7,color=C['ro'],ha='center')
fig.savefig(OUT+'cap3/cap3_pipeline.pdf'); fig.savefig(OUT+'cap3/cap3_pipeline.png'); plt.close(fig)

# ---------- F3 C3D canonico ----------
fig,ax=canvas(6.4,2.6)
box(ax,0.00,0.05,0.30,0.88,'POINT','18 centri articolari\n(coordinate in metri)\n\nun canale per giunto\ncon gli angoli su\nsagittale, frontale,\ntrasverso',fc='#f3f3f3')
box(ax,0.33,0.05,0.30,0.88,'ANALOG','per ogni segmento:\nquaternione,\nvelocità angolare,\naccelerazione angolare',fc='#f3f3f3')
box(ax,0.66,0.05,0.34,0.88,'BIOMECH (proprietario)','VERSION, CONVENTION\nSOURCE, SOURCE_FORMAT,\nCAPTURE_SYSTEM, SPORT\n\nPOINT_PROVENANCE,\nPOINT_COVERAGE\nSEGMENT_PROVENANCE\nANGLE_PROVENANCE\nANGLE_IMPLAUSIBLE_RATE\n\nABSENT, ABSENT_REASON,\nABSENT_DETAIL\n\nCYCLE_METHOD, CYCLE_SIDE,\nCYCLE_START, CYCLE_END,\nCYCLE_WARNINGS',fc='#fde2cf',bsize=7,lw=1.6)
ax.text(0.315,0.965,'gruppi standard C3D, leggibili da qualunque strumento',ha='center',fontsize=7.5,color='0.3')
ax.text(0.83,0.965,'contratto di risposta',ha='center',fontsize=7.5,color=C['ro'])
fig.savefig(OUT+'cap3/cap3_c3d_canonico.pdf'); fig.savefig(OUT+'cap3/cap3_c3d_canonico.png'); plt.close(fig)

# ---------- F5 schema DB ----------
fig,ax=canvas(6.8,4.0)
# tre colonne di riquadri larghi BW, con un intervallo di 0.155 fra una colonna e l'altra
BW=0.23; X0,X1,X2=0.00,0.385,0.77
box(ax,X0,0.70,BW,0.28,'Vocabolario','provenance_kinds\nabsence_reasons\nplanes, sports\npoints, segments, joints',fc='#f3f3f3')
box(ax,X0,0.36,BW,0.26,'subjects_identity','anagrafica separata:\nnessun ruolo\napplicativo la legge',fc='#f7e3ef')
box(ax,X1,0.70,BW,0.28,'sessions','chiave, sport, fps,\nformato, convenzione,\nversione, pseudonimo',fc='#dcecf8',lw=1.6)
box(ax,X2,0.70,BW,0.28,'Serie','point_trajectories\nsegment_series\njoint_angles\nabsences',fc='#dcecf8')
box(ax,X1,0.36,BW,0.26,'cycles','inizio, fine,\nmetodo, lato, avvisi',fc='#dcecf8')
box(ax,X2,0.36,BW,0.26,'Per ciclo','cycle_metrics\ncycle_atoms:\nvector(32), indice HNSW\ncoseno, versione encoder',fc='#dcf1e8')
box(ax,X1,0.02,BW,0.26,'alerts','osservato, atteso,\nscostamento, gravità,\ncopertura, provenienza',fc='#fde2cf')
box(ax,X0,0.02,BW,0.26,'reference_ranges','76 intervalli\nambito, gesto, giunto,\npiano, metrica, fonte',fc='#fde2cf')
box(ax,X2,0.02,BW,0.26,'Viste','session_signals\nsession_absences\nsession_atoms\nsession_alerts',fc='#f3f3f3')
cx=X1+BW/2; g=0.012  # margine: il bordo arrotondato del riquadro sporge di 0.01
arr(ax,(X1+BW+g,0.84),(X2-g,0.84)); arr(ax,(cx,0.70-g),(cx,0.62+g)); arr(ax,(X1+BW+g,0.49),(X2-g,0.49))
arr(ax,(cx,0.36-g),(cx,0.28+g)); arr(ax,(X0+BW+g,0.15),(X1-g,0.15)); arr(ax,(X0+BW+g,0.84),(X1-g,0.84),'chiavi esterne',off=(0,0.02))
ax.plot([X0+BW/2,X0+BW/2],[0.62,0.70],ls=':',color='0.5'); ax.text(X0+BW/2+0.005,0.655,' solo pseudonimo',fontsize=6.5,color='0.4')
fig.savefig(OUT+'cap3/cap3_database.pdf'); fig.savefig(OUT+'cap3/cap3_database.png'); plt.close(fig)

# ---------- F6 messa in opera ----------
fig,ax=canvas(6.8,3.4)
# colonne: client, server e sottoprocessi, Docker; gli intervalli lasciano spazio alle etichette delle frecce
XC,WC=0.015,0.17; XS,WS=0.265,0.22; XD,WD=0.635,0.35; XB,WB=0.665,0.29; g=0.012
ax.add_patch(FancyBboxPatch((0.0,0.0),1.0,0.93,boxstyle='round,pad=0.005,rounding_size=0.02',fc='none',ec='0.5',ls='--'))
ax.text(0.01,0.955,'macchina locale',fontsize=8,color='0.3')
ax.add_patch(FancyBboxPatch((XD,0.05),WD,0.80,boxstyle='round,pad=0.005,rounding_size=0.02',fc='#eef6fb',ec='0.5'))
ax.text(XD+0.01,0.815,'Docker Compose',fontsize=8,color='0.3')
box(ax,XB,0.44,WB,0.32,'db: pgvector/pgvector:pg16','127.0.0.1:5432, TLS\nschema.sql e setup_roles.sql\nal primo avvio del volume',fc='#dcecf8',tsize=8.5)
box(ax,XB,0.09,WB,0.28,'ingest (profilo tools)','conversione e caricamento\nda riga di comando',fc='#fff3dc')
box(ax,XC,0.52,WC,0.30,'Client MCP','agente\nconversazionale',fc='#f3f3f3')
box(ax,XS,0.52,WS,0.30,'Server MCP','python -m src.mcp.server\nruolo biomech_mcp\n(sola lettura)',fc='#dcf1e8')
box(ax,XS,0.09,WS,0.33,'Sottoprocessi','conversione e caricamento\n(ruolo biomech_ingest)\n\nvisualizzatore Open3D\n(ambiente separato)',fc='#f7e3ef',bsize=7)
gm=(XS+WS+XD)/2  # centro dell'intervallo fra server e Docker
arr(ax,(XC+WC+g,0.67),(XS-g,0.67)); ax.text((XC+WC+XS)/2,0.685,'stdio',fontsize=7,color='0.25',ha='center',va='bottom')
arr(ax,(XS+WS+g,0.67),(XB-g,0.62)); ax.text(gm,0.61,'SQL\nsu TLS',fontsize=7,color='0.25',ha='center',va='top')
cx=XS+WS/2
arr(ax,(cx,0.52-g),(cx,0.42+g)); ax.text(cx+0.012,0.465,'avvia',fontsize=7,color='0.25',va='center')
arr(ax,(XS+WS+g,0.28),(XB-g,0.50)); ax.text(gm,0.27,'scrittura',fontsize=7,color='0.25',ha='center',va='top')
arr(ax,(XB+WB/2,0.37+g),(XB+WB/2,0.44-g))
fig.savefig(OUT+'cap3/cap3_messa_in_opera.pdf'); fig.savefig(OUT+'cap3/cap3_messa_in_opera.png'); plt.close(fig)
print('ok')
