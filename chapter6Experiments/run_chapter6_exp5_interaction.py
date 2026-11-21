#!/usr/bin/env python3
"""
run_chapter6_exp5_interaction.py — ARSPI-Net Chapter 6, Experiment 6.5
Diagnosis × Subcategory Interaction in Reservoir Dynamics

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

MATHEMATICAL MOTIVATION:
  Experiment 6.4 established that validated dynamical metrics dissociate
  fine-grained affective subcategories (Threat vs Mutilation d=0.31,
  Cute vs Erotic d=0.40). The next question: does clinical status
  modulate these category-specific dynamical signatures?

  If MDD/SUD/PTSD alter the reservoir's temporal dynamics DIFFERENTLY
  for Threat vs Cute, that is a diagnosis × category interaction —
  evidence that psychopathology disrupts specific affective processing channels.

EXPERIMENTAL DESIGN:
  5 diagnosis variables (MDD n=142, SUD n=85, PTSD n=82, GAD n=60, ADHD n=50)
  × 4 subcategories × 9 validated metrics.
  Interaction tested via permutation (500 iterations) on category reactivity.

Generates 4 figures:
  raw6_5a_sud_population_rates.pdf — SUD+/- pop rate traces × 4 categories
  raw6_5b_diag_category_matrix.pdf — 4 metrics × 5 diagnoses × 4 categories
  raw6_5c_reactivity_profiles.pdf  — Category reactivity by diagnosis group
  analysis6_5d_interaction.pdf     — Interaction heatmap + strongest effects

Results:
  SUD: category-dependent hypoactivation (Mutilation d=-0.46, Cute d=-0.05)
  ADHD: global hyperactivation (d=+0.40 for negative, category-independent)
  MDD: selective positive-arousal enhancement (Erotic d=+0.22)

Usage:
    python run_chapter6_exp5_interaction.py \
        --category-dirs categoriesbatch1 ... \
        --psychopathology SHAPE_Community_Andrew_Psychopathology.xlsx \
        --exp4-results results/ch6_exp4_full.pkl \
        --output-dir figures/ --results-dir results/

Author: Andrew (ARSPI-Net Dissertation), March 2026
"""
import numpy as np, os, re, pickle, time, math, argparse
from collections import defaultdict
from scipy.stats import mannwhitneyu
from scipy.ndimage import uniform_filter1d
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N_RES=256; BETA=0.05; M_TH=0.5; EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}
DIAGNOSES=['MDD','SUD','PTSD','GAD','ADHD']
DCOL={'MDD':'#9b59b6','SUD':'#e67e22','PTSD':'#2c3e50','GAD':'#1abc9c','ADHD':'#3498db'}
VALIDATED=['total_spikes','mean_firing_rate','phi_proxy','temporal_sparsity',
           'permutation_entropy','tau_relax','tau_ac','rate_entropy','rate_variance']

class Res:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed);self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float);np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask;self.fanout=mask.sum(1)
    def run(self,u):
        T=len(u);M=np.zeros((T,N_RES));S=np.zeros((T,N_RES),dtype=np.int8)
        m=np.zeros(N_RES);s=np.zeros(N_RES)
        for t in range(T):
            I=self.Win*u[t]+self.Wrec@s;m=(1-BETA)*m*(1-s)+I
            sp=(m>=M_TH).astype(float);M[t]=m;S[t]=sp.astype(np.int8);s=sp
        return M,S

def load_files(dirs):
    af={}
    for d in dirs:
        if not os.path.isdir(d):continue
        for f in os.listdir(d):
            m=PAT.match(f)
            if m:
                sid=int(m.group(1));cat=m.group(3)
                if sid not in EXCLUDED:af[(sid,cat)]=os.path.join(d,f)
    return af

def norm(u):return(u-u.mean())/(u.std()+1e-10)

def main():
    p=argparse.ArgumentParser(description='Exp 6.5: Diagnosis x Category Interaction')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--psychopathology',type=str,required=True,
                   help='Path to SHAPE_Community_Andrew_Psychopathology.xlsx')
    p.add_argument('--exp4-results',type=str,required=True,
                   help='Path to ch6_exp4_full.pkl from Experiment 6.4')
    p.add_argument('--output-dir',default='figures')
    p.add_argument('--results-dir',default='results')
    p.add_argument('--n-permutations',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time()

    print("="*70)
    print("EXPERIMENT 6.5: DIAGNOSIS x SUBCATEGORY INTERACTION")
    print("="*70)

    # Load Experiment 6.4 metrics
    d4=pickle.load(open(a.exp4_results,'rb'))
    all_mets=d4['all_metrics'];subjects=d4['subjects'];channels=d4['channels']

    # Build subject-level metrics (average across channels)
    scm={}
    for sid in subjects:
        for cat in CATS:
            vals={m:[] for m in VALIDATED}
            for ch in channels:
                k=(sid,cat,ch)
                if k in all_mets:
                    for m_ in VALIDATED:vals[m_].append(all_mets[k][m_])
            scm[(sid,cat)]={m_:np.mean(vals[m_]) if vals[m_] else np.nan for m_ in VALIDATED}

    # Load clinical metadata
    psych=pd.read_excel(a.psychopathology)
    valid_ids=set(subjects)
    psych=psych[psych['ID'].isin(valid_ids)]
    pd_=psych.set_index('ID').to_dict('index')
    print(f"Loaded {len(psych)} subjects with clinical metadata")

    for diag in DIAGNOSES:
        n1=sum(1 for s in subjects if s in pd_ and pd_[s].get(diag,0)==1)
        n0=sum(1 for s in subjects if s in pd_ and pd_[s].get(diag,0)==0)
        print(f"  {diag}: +={n1}, -={n0}")

    # Load EEG files for raw observation figures
    af=load_files(a.category_dirs)
    reservoir=Res(seed=a.seed)

    # ── FIG 1: Raw Observation — SUD pop rates × 4 categories ──
    dch=16
    sud_pos=[s for s in subjects if s in pd_ and pd_[s].get('SUD',0)==1]
    sud_neg=[s for s in subjects if s in pd_ and pd_[s].get('SUD',0)==0]

    fig=plt.figure(figsize=(20,14));gs=GridSpec(2,4,figure=fig,hspace=0.4,wspace=0.3)
    rng_s=np.random.RandomState(a.seed)
    sp_sample=rng_s.choice(sud_pos,min(15,len(sud_pos)),replace=False)
    sn_sample=rng_s.choice(sud_neg,min(15,len(sud_neg)),replace=False)

    for ci,cat in enumerate(CATS):
        # SUD-
        ax=fig.add_subplot(gs[0,ci])
        for sid in sn_sample:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);_,S=reservoir.run(u)
            ax.plot(uniform_filter1d(S.mean(1).astype(float),30),color=CCOL[cat],lw=0.3,alpha=0.3)
        grand=[]
        for sid in sud_neg[:50]:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);_,S=reservoir.run(u)
            grand.append(S.mean(1).astype(float))
        if grand:ax.plot(uniform_filter1d(np.mean(grand,0),30),'k-',lw=2.5)
        ax.set_title(f'{cat}',fontsize=11,fontweight='bold',color=CCOL[cat])
        ax.set_ylabel('SUD-\nPop Rate' if ci==0 else '',fontsize=10,fontweight='bold')
        ax.set_ylim(0.05,0.25);ax.grid(alpha=0.1)
        # SUD+
        ax=fig.add_subplot(gs[1,ci])
        for sid in sp_sample:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);_,S=reservoir.run(u)
            ax.plot(uniform_filter1d(S.mean(1).astype(float),30),color=CCOL[cat],lw=0.3,alpha=0.3)
        grand=[]
        for sid in sud_pos[:50]:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);_,S=reservoir.run(u)
            grand.append(S.mean(1).astype(float))
        if grand:ax.plot(uniform_filter1d(np.mean(grand,0),30),'k-',lw=2.5)
        ax.set_ylabel('SUD+\nPop Rate' if ci==0 else '',fontsize=10,fontweight='bold')
        ax.set_xlabel('Time Step');ax.set_ylim(0.05,0.25);ax.grid(alpha=0.1)

    fig.suptitle(f'Raw Observation 6.14: Population Rate by SUD Status x Category\n'
                 f'Top: SUD- (n={len(sud_neg)}). Bottom: SUD+ (n={len(sud_pos)}). '
                 f'Thin=individual, Black=grand average.',
                 fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/raw6_5a_sud_population_rates.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_5a ({time.time()-t0:.0f}s)")

    # ── FIG 2: Raw Observation — Metric × Diagnosis × Category ──
    key_mets=['permutation_entropy','tau_ac','total_spikes','rate_entropy']
    key_labs=['H_pi','tau_AC','Spikes','H_rate']
    fig,axes=plt.subplots(4,5,figsize=(24,18))
    for mi,(met,lab) in enumerate(zip(key_mets,key_labs)):
        for di,diag in enumerate(DIAGNOSES):
            ax=axes[mi,di]
            dp=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==1]
            dn=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==0]
            positions=[];data_b=[];col_b=[]
            for ci,cat in enumerate(CATS):
                vn=[scm[(s,cat)][met] for s in dn if (s,cat) in scm and not np.isnan(scm[(s,cat)][met])]
                vp=[scm[(s,cat)][met] for s in dp if (s,cat) in scm and not np.isnan(scm[(s,cat)][met])]
                pb=ci*3;data_b.extend([vn,vp]);positions.extend([pb,pb+1]);col_b.extend([CCOL[cat]]*2)
            bp=ax.boxplot(data_b,positions=positions,widths=0.7,patch_artist=True,showfliers=False)
            for pi_,(patch,col) in enumerate(zip(bp['boxes'],col_b)):
                patch.set_facecolor(col);patch.set_alpha(0.3 if pi_%2==0 else 0.7)
            ax.set_xticks([0.5,3.5,6.5,9.5]);ax.set_xticklabels(['Thr','Mut','Cut','Ero'],fontsize=7)
            if mi==0:
                np_=len(dp);nn_=len(dn)
                ax.set_title(f'{diag} (+:{np_} -:{nn_})',fontsize=10,fontweight='bold',color=DCOL[diag])
            if di==0:ax.set_ylabel(lab,fontsize=10,fontweight='bold')
            ax.grid(axis='y',alpha=0.1)
    fig.suptitle('Raw Observation 6.15: Metrics x Diagnoses x Categories\nLight=Diag-, Dark=Diag+',
                 fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_5b_diag_category_matrix.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_5b ({time.time()-t0:.0f}s)")

    # ── FIG 3: Raw Observation — Reactivity Profiles ──
    fig,axes=plt.subplots(len(DIAGNOSES),len(key_mets),figsize=(20,20))
    for di,diag in enumerate(DIAGNOSES):
        dp=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==1]
        dn=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==0]
        for mi,(met,lab) in enumerate(zip(key_mets,key_labs)):
            ax=axes[di,mi]
            for grp,gsids,col,lbl,ls in [('neg',dn,'#3498db',f'{diag}-','-'),('pos',dp,'#e74c3c',f'{diag}+','--')]:
                react={cat:[] for cat in CATS}
                for sid in gsids:
                    cv=[scm[(sid,cat)][met] for cat in CATS if (sid,cat) in scm]
                    if len(cv)!=4:continue
                    sm=np.mean(cv)
                    for ci,cat in enumerate(CATS):react[cat].append(cv[ci]-sm)
                means=[np.mean(react[cat]) for cat in CATS]
                sems=[np.std(react[cat])/np.sqrt(len(react[cat])+1e-10) for cat in CATS]
                ax.errorbar(range(4),means,yerr=sems,color=col,lw=1.5,ls=ls,marker='o',ms=5,capsize=3,label=lbl)
            ax.set_xticks(range(4));ax.set_xticklabels(CATS,fontsize=7,rotation=20)
            ax.axhline(0,color='gray',lw=0.5,ls=':');ax.grid(axis='y',alpha=0.1)
            if di==0:ax.set_title(lab,fontsize=11,fontweight='bold')
            if mi==0:ax.set_ylabel(f'{diag}',fontsize=11,fontweight='bold',color=DCOL[diag])
            if di==0 and mi==0:ax.legend(fontsize=7)
    fig.suptitle('Raw Observation 6.16: Category Reactivity by Diagnosis\nCrossing lines=interaction. Parallel=main effect only.',
                 fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_5c_reactivity_profiles.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_5c ({time.time()-t0:.0f}s)")

    # ── ANALYSIS: Permutation-tested interactions ──
    print("\n  Computing interaction tests...")
    interaction_results={}
    for diag in DIAGNOSES:
        interaction_results[diag]={}
        dp=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==1]
        dn=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==0]
        for met in VALIDATED:
            cat_ds={}
            for cat in CATS:
                vp=[scm[(s,cat)][met] for s in dp if (s,cat) in scm]
                vn=[scm[(s,cat)][met] for s in dn if (s,cat) in scm]
                pool=np.sqrt((np.std(vp)**2+np.std(vn)**2)/2)
                cat_ds[cat]=(np.mean(vp)-np.mean(vn))/(pool+1e-15)
            # Reactivity-based interaction test
            rp=[];rn=[]
            for sid in dp:
                cv=[scm[(sid,cat)][met] for cat in CATS if (sid,cat) in scm]
                if len(cv)==4:sm=np.mean(cv);rp.append([cv[i]-sm for i in range(4)])
            for sid in dn:
                cv=[scm[(sid,cat)][met] for cat in CATS if (sid,cat) in scm]
                if len(cv)==4:sm=np.mean(cv);rn.append([cv[i]-sm for i in range(4)])
            rp=np.array(rp);rn=np.array(rn)
            if len(rp)>5 and len(rn)>5:
                diff_profile=rp.mean(0)-rn.mean(0)
                F_obs=np.var(diff_profile)
                all_r=np.vstack([rp,rn]);np_=len(rp)
                rng_p=np.random.RandomState(a.seed);F_null=[]
                for _ in range(a.n_permutations):
                    pi_=rng_p.permutation(len(all_r))
                    dp_=all_r[pi_[:np_]].mean(0)-all_r[pi_[np_:]].mean(0)
                    F_null.append(np.var(dp_))
                # (1 + count) / (1 + n): a permutation p-value is never exactly zero.
                # Fixed 2026-11 (docs/v1/DEFECT_FIXES.md, D4).
                p_int=(1+np.sum(np.array(F_null)>=F_obs))/(1+len(F_null))
            else:F_obs=0;p_int=1.0
            interaction_results[diag][met]={'cat_ds':cat_ds,'F_interact':float(F_obs),
                                            'p_interact':float(p_int),'n_pos':len(rp),'n_neg':len(rn)}

    # Print significant
    print(f"\n  {'Diag':<6} {'Metric':<22} {'d_Thr':>7} {'d_Mut':>7} {'d_Cut':>7} {'d_Ero':>7} {'p_int':>8}")
    print("  "+"-"*65)
    for diag in DIAGNOSES:
        for met in VALIDATED:
            r=interaction_results[diag][met];ds=r['cat_ds']
            if r['p_interact']<0.1 or any(abs(v)>0.2 for v in ds.values()):
                print(f"  {diag:<6}{met:<20}{ds['Threat']:+7.3f}{ds['Mutilation']:+7.3f}"
                      f"{ds['Cute']:+7.3f}{ds['Erotic']:+7.3f}{r['p_interact']:8.4f}")

    # ── FIG 4: Analysis Figure ──
    fig=plt.figure(figsize=(22,16));gs=GridSpec(3,5,figure=fig,hspace=0.55,wspace=0.4)
    # Row 1: Per-diagnosis effect size heatmaps
    for di,diag in enumerate(DIAGNOSES):
        ax=fig.add_subplot(gs[0,di])
        dm=np.zeros((len(VALIDATED),4))
        for mi,met in enumerate(VALIDATED):
            for ci,cat in enumerate(CATS):dm[mi,ci]=interaction_results[diag][met]['cat_ds'][cat]
        im=ax.imshow(dm,aspect='auto',cmap='RdBu_r',vmin=-0.3,vmax=0.3)
        ax.set_xticks(range(4));ax.set_xticklabels(['Thr','Mut','Cut','Ero'],fontsize=7)
        short=[m.replace('_',' ')[:10] for m in VALIDATED]
        ax.set_yticks(range(len(VALIDATED)));ax.set_yticklabels(short if di==0 else [],fontsize=6)
        np_=len([s for s in subjects if s in pd_ and pd_[s].get(diag,0)==1])
        ax.set_title(f'{diag} (+:{np_})',fontsize=10,fontweight='bold',color=DCOL[diag])
        for mi,met in enumerate(VALIDATED):
            if interaction_results[diag][met]['p_interact']<0.05:
                ax.text(3.6,mi,'*',fontsize=12,fontweight='bold',va='center')
    # Row 2: p-value matrix
    ax=fig.add_subplot(gs[1,0:3])
    pm=np.zeros((len(VALIDATED),len(DIAGNOSES)))
    for di,diag in enumerate(DIAGNOSES):
        for mi,met in enumerate(VALIDATED):pm[mi,di]=interaction_results[diag][met]['p_interact']
    lp=-np.log10(pm+1e-10)
    im=ax.imshow(lp,aspect='auto',cmap='YlOrRd',vmin=0,vmax=3)
    ax.set_xticks(range(len(DIAGNOSES)));ax.set_xticklabels(DIAGNOSES,fontsize=9)
    ax.set_yticks(range(len(VALIDATED)));ax.set_yticklabels([m.replace('_',' ')[:15] for m in VALIDATED],fontsize=7)
    plt.colorbar(im,ax=ax,shrink=0.8,label='-log10(p)')
    for mi in range(len(VALIDATED)):
        for di in range(len(DIAGNOSES)):
            if pm[mi,di]<0.05:ax.text(di,mi,f'{pm[mi,di]:.3f}',ha='center',va='center',fontsize=6,fontweight='bold')
    ax.set_title(f'Interaction p-values (permutation, {a.n_permutations} iter)',fontsize=11,fontweight='bold')
    # Row 2 right: summary
    ax=fig.add_subplot(gs[1,3:5])
    sc=np.sum(pm<0.05,axis=1)
    ax.barh(range(len(VALIDATED)),sc,color=['#27ae60' if c>0 else '#95a5a6' for c in sc],edgecolor='black',lw=0.3)
    ax.set_yticks(range(len(VALIDATED)));ax.set_yticklabels([m.replace('_',' ')[:15] for m in VALIDATED],fontsize=7)
    ax.set_xlabel('# Diagnoses with Interaction (p<0.05)');ax.set_title('Metric Sensitivity',fontsize=11,fontweight='bold')
    # Row 3: top 5 interactions
    bp_=[];
    for diag in DIAGNOSES:
        for met in VALIDATED:
            r=interaction_results[diag][met];bp_.append((r['p_interact'],diag,met,r))
    bp_.sort()
    for pi in range(min(5,len(bp_))):
        pv,diag,met,r=bp_[pi];ax=fig.add_subplot(gs[2,pi])
        dp=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==1]
        dn=[s for s in subjects if s in pd_ and pd_[s].get(diag,0)==0]
        for gsids,col,lbl,mk in [(dn,'#3498db',f'{diag}-','o'),(dp,'#e74c3c',f'{diag}+','s')]:
            means=[];sems=[]
            for cat in CATS:
                vals=[scm[(s,cat)][met] for s in gsids if (s,cat) in scm]
                means.append(np.mean(vals));sems.append(np.std(vals)/np.sqrt(len(vals)+1e-10))
            ax.errorbar(range(4),means,yerr=sems,color=col,lw=1.5,marker=mk,ms=5,capsize=3,label=lbl)
        ax.set_xticks(range(4));ax.set_xticklabels(CATS,fontsize=7,rotation=20)
        ax.set_title(f'{diag} x {met[:12]}\np={pv:.4f}',fontsize=9,fontweight='bold')
        ax.legend(fontsize=6);ax.grid(axis='y',alpha=0.1)
    fig.suptitle('Analysis 6.5: Diagnosis x Subcategory Interaction\n'
                 f'5 diagnoses x 9 metrics x 4 categories. Permutation-tested ({a.n_permutations} iter).',
                 fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_5d_interaction.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis6_5d ({time.time()-t0:.0f}s)")

    pickle.dump({'interaction_results':interaction_results,'diagnoses':DIAGNOSES,
                 'validated_metrics':VALIDATED,'categories':CATS,'subjects':subjects},
                open(f'{a.results_dir}/ch6_exp5_full.pkl','wb'))
    print(f"\nDone. 4 figures. {time.time()-t0:.0f}s.")

if __name__=='__main__':main()
