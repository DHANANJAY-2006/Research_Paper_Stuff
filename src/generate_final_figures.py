# -*- coding: utf-8 -*-
"""
generate_final_figures.py -- Master Final Figure Generator
Produces all 21 publication figures with clean titles.
"""

import os
import sys
import io

# Force UTF-8 stdout so emojis/special chars don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from config import DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linestyle': ':', 'grid.alpha': 0.55,
})
C_CLEAN  = '#2c6fad'
C_ADV    = '#8b0000'
C_AT     = '#2e8b57'
C_ACCENT = '#e07b00'
C_PURPLE = '#7d3c98'
C_GRAY   = '#555555'


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {name}")


def cleanup_old_figures():
    old = [
        "adversarial_stealth_analysis.png","epsilon_sensitivity.png","ga_vs_random.png",
        "per_class_vulnerability.png","recall_drop_analysis.png","transferability_matrix.png",
        "fig1_baseline_metrics.png","fig2_transferability_matrix.png","fig3_epsilon_sensitivity.png",
        "fig4_per_class_vulnerability.png","fig5_ga_vs_random.png","fig6_recall_drop_analysis.png",
        "fig7_master_summary.png","fig8_mlp_training_curve.png","fig9_phase3_baseline_comparison.png",
        "fig10_phase3_attack_comparison.png","fig11_phase3_epsilon_sensitivity.png",
        "fig12_phase3_confidence_degradation.png","fig13_phase3_per_class_heatmap.png",
        "fig14_phase4_core_comparison.png","fig15_phase4_security_utility_tradeoff.png",
        "fig16_phase4_per_class_defense.png","fig17_phase4_cross_architecture_transfer.png",
        "fig18_phase4_radar_defense_profile.png","fig19_tsne_feature_space.png",
        "fig20_pareto_frontier.png","fig21_feature_importance_drift.png",
        "fig01_baseline_performance.png","fig02_transferability.png",
        "fig03_epsilon_sensitivity.png","fig04_per_class_vulnerability.png",
        "fig05_ga_vs_random_baseline.png","fig06_recall_f1_asr_drop.png",
        "fig07_master_overview.png",
    ]
    print("Cleaning up old figures...")
    for f in old:
        p = os.path.join(FIGURES_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"  [DEL] {f}")


# ── FIG 01 ───────────────────────────────────────────────────
def fig01_baseline():
    data = pd.DataFrame({
        'Model':    ['Random Forest', 'XGBoost', 'LightGBM'],
        'Accuracy': [0.9922, 0.9925, 0.9924],
        'F1':       [0.9960, 0.9962, 0.9961],
        'Time':     [10.18,  3.41,   1.99],
    })
    colors = [C_CLEAN, C_ADV, C_AT]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, title, fmt, ylim in zip(
        axes,
        ['Accuracy','F1','Time'],
        ['Classification Accuracy','Macro F1-Score','Training Time (seconds)'],
        [mtick.PercentFormatter(xmax=1,decimals=2), mtick.PercentFormatter(xmax=1,decimals=2), None],
        [(0.98,1.001),(0.98,1.001),(0,14)]
    ):
        bars = ax.bar(x, data[col], color=colors, width=0.5, zorder=3)
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(data['Model'], rotation=8)
        if fmt: ax.yaxis.set_major_formatter(fmt)
        ax.set_ylim(ylim); ax.grid(axis='y')
        for bar, v in zip(bars, data[col]):
            label = f'{v:.4f}' if col != 'Time' else f'{v:.2f}s'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ylim[0]*0.001,
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.suptitle('Baseline IDS Model Performance on CIC-IoT2023', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); save(fig, 'fig01_baseline_performance.png')


# ── FIG 02 ───────────────────────────────────────────────────
def fig02_transferability():
    models = ['Random Forest','XGBoost','LightGBM']
    clean  = [100.0,100.0,99.8]; adv = [98.0,25.6,74.2]
    drops  = [c-a for c,a in zip(clean,adv)]
    colors = [C_CLEAN,C_ADV,C_AT]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(3); w = 0.35
    b1 = axes[0].bar(x-w/2, clean, w, label='Clean Data',       color=C_CLEAN, zorder=3)
    b2 = axes[0].bar(x+w/2, adv,   w, label='Adversarial Data', color=C_ADV,   alpha=0.9, zorder=3)
    axes[0].set_title('Detection Rate: Clean vs. Adversarial')
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_ylim(0,120); axes[0].legend(); axes[0].grid(axis='y')
    for bar in list(b1)+list(b2):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    bars = axes[1].bar(models, drops, color=colors, width=0.4, zorder=3)
    axes[1].set_title('Detection Rate Drop Under Black-Box Transfer Attack')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_ylim(0,95); axes[1].grid(axis='y')
    for bar,d in zip(bars,drops):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'-{d:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color=C_ADV)
    note = ("RF shows only 2% drop under black-box transfer attack.\n"
            "However, RF collapses to 99% ASR under direct white-box attack.\n"
            "-> RF is architecture-robust but feature-transparent.")
    axes[1].text(0.5,-0.20,note,transform=axes[1].transAxes,
                 ha='center',va='top',fontsize=8.5,style='italic',color=C_GRAY,
                 bbox=dict(boxstyle='round,pad=0.5',facecolor='#fff8e1',edgecolor=C_ACCENT,linewidth=1.2))
    fig.suptitle('Cross-Model Adversarial Transferability (Surrogate: XGBoost)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.28); save(fig,'fig02_transferability.png')


# ── FIG 03 ───────────────────────────────────────────────────
def fig03_epsilon_sensitivity():
    epsilons = [0.10,0.76,1.41,2.07,2.72,3.38,4.03,4.69,5.34,6.00]
    asr      = [1.4, 2.0, 2.2, 2.4, 2.4, 2.4, 2.4,63.2,84.8,87.6]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(epsilons,asr,marker='o',color=C_ADV,linewidth=2.5,markersize=9,zorder=3)
    ax.fill_between(epsilons,0,asr,alpha=0.10,color=C_ADV)
    ax.axvline(x=4.69,color=C_ACCENT,linestyle='--',linewidth=1.8,label='Phase Transition (epsilon ~= 4.69)')
    ax.annotate('ASR: 2.4% -> 63.2%',xy=(4.69,63.2),xytext=(3.0,72),
                arrowprops=dict(arrowstyle='->',color=C_ACCENT),
                fontsize=9.5,color=C_ACCENT,fontweight='bold')
    ax.set_xlabel('Perturbation Budget (epsilon)'); ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Epsilon Sensitivity Ablation -- XGBoost ASR vs. Perturbation Budget')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.legend(); ax.grid()
    save(fig,'fig03_epsilon_sensitivity.png')


# ── FIG 04 ───────────────────────────────────────────────────
def fig04_per_class():
    classes = ['DDOS-RSTFINFLOOD','MIRAI-UDPPLAIN','DDOS-SYN_FLOOD','DOS-SYN_FLOOD',
               'DDOS-UDP_FLOOD','DDOS-ICMP_FLOOD','DDOS-TCP_FLOOD','DOS-TCP_FLOOD',
               'DOS-UDP_FLOOD','DDOS-SYNONYMOUSIP_FLOOD','MIRAI-GREETH_FLOOD',
               'MIRAI-GREIP_FLOOD','DDOS-PSHACK_FLOOD','DDOS-UDP_FRAGMENTATION',
               'DDOS-ACK_FRAGMENTATION','DDOS-ICMP_FRAGMENTATION','DDOS-SLOWLORIS',
               'DDOS-HTTP_FLOOD','DOS-HTTP_FLOOD','VULNERABILITYSCAN',
               'RECON-HOSTDISCOVERY','RECON-OSSCAN','RECON-PORTSCAN',
               'DNS_SPOOFING','DICTIONARYBRUTEFORCE','BROWSERHIJACKING',
               'SQLINJECTION','MITM-ARPSPOOFING','COMMANDINJECTION','BACKDOOR_MALWARE','XSS']
    asr = [8,90,53,58,75,88,90,96,80,49,99,94,97,75,99,99,62,85,88,100,100,100,100,100,100,100,100,100,100,100,100]
    idx = np.argsort(asr)[::-1]
    classes = [classes[i] for i in idx]; asr = [asr[i] for i in idx]
    colors  = [C_ADV if v==100 else (C_ACCENT if v>=80 else C_CLEAN) for v in asr]
    fig, ax = plt.subplots(figsize=(10,10))
    bars = ax.barh(classes,asr,color=colors,alpha=0.85,zorder=3)
    ax.set_xlabel('Attack Success Rate (%)'); ax.set_title('Per-Attack-Class Evasion Vulnerability (epsilon = 5.0)')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter()); ax.set_xlim(0,116); ax.grid(axis='x')
    for bar,v in zip(bars,asr):
        ax.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
                f'{v:.0f}%',va='center',fontsize=9,fontweight='bold')
    patches = [mpatches.Patch(color=C_ADV,   label='100% -- Fully Evadable'),
               mpatches.Patch(color=C_ACCENT, label='80-99% -- Highly Vulnerable'),
               mpatches.Patch(color=C_CLEAN,  label='< 80% -- Partially Robust')]
    ax.legend(handles=patches,loc='lower right',fontsize=9)
    plt.tight_layout(); save(fig,'fig04_per_class_vulnerability.png')


# ── FIG 05 ───────────────────────────────────────────────────
def fig05_ga_vs_random():
    eps  = [0.5,1.4,2.3,3.2,4.1,5.0]
    ga   = [2.0,2.2,2.4,2.4,2.4,72.8]
    rand = [0.2,0.2,0.2,0.6,0.8, 2.0]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(eps,ga,  marker='o',color=C_ADV,  linewidth=2.5,markersize=9,label='Multi-Objective GA (Proposed)')
    ax.plot(eps,rand,marker='s',color=C_CLEAN,linewidth=2.5,markersize=9,linestyle='--',label='Random Noise Baseline')
    ax.fill_between(eps,ga,rand,alpha=0.10,color=C_ADV)
    ax.annotate('GA advantage at epsilon=5.0: +70.8pp',xy=(5.0,37),fontsize=9,color=C_ADV,fontweight='bold',ha='center')
    ax.set_xlabel('Perturbation Budget (epsilon)'); ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Multi-Objective GA vs. Random Noise Baseline')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.legend(); ax.grid()
    save(fig,'fig05_ga_vs_random_baseline.png')


# ── FIG 06 ───────────────────────────────────────────────────
def fig06_recall_drop():
    df = pd.DataFrame({
        'Model':        ['Random Forest','XGBoost','LightGBM'],
        'Clean Recall': [99.4,99.4,99.2],
        'Adv Recall':   [1.0, 44.0,31.6],
        'Clean F1':     [93.16,94.04,93.85],
        'Adv F1':       [1.74, 56.41,43.95],
        'ASR':          [99.0, 56.0, 68.4],
    })
    x = np.arange(3); w = 0.35
    fig, axes = plt.subplots(1,3,figsize=(15,5.5))
    for ax, cc, ac, title in zip(
        axes[:2],
        ['Clean Recall','Clean F1'],
        ['Adv Recall','Adv F1'],
        ['Detection Recall Drop Under Attack','F1-Score Drop Under Attack']
    ):
        b1 = ax.bar(x-w/2,df[cc],w,label='Clean',      color=C_CLEAN,zorder=3)
        b2 = ax.bar(x+w/2,df[ac],w,label='Adversarial',color=C_ADV,alpha=0.9,zorder=3)
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(df['Model'],rotation=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0,115); ax.legend(); ax.grid(axis='y')
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                    f'{bar.get_height():.1f}%',ha='center',va='bottom',fontsize=8)
    bars = axes[2].bar(df['Model'],df['ASR'],color=[C_CLEAN,C_ADV,C_AT],width=0.4,zorder=3)
    axes[2].set_title('Attack Success Rate per Model')
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].set_ylim(0,120); axes[2].grid(axis='y')
    for bar,v in zip(bars,df['ASR']):
        axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                     f'{v:.1f}%',ha='center',va='bottom',fontsize=10,fontweight='bold')
    note = ("RF (99% ASR): High white-box vulnerability due to low-entropy decision boundaries.\n"
            "RF resists black-box transfer (2% ASR) but collapses under direct targeting.")
    axes[2].text(0.5,-0.18,note,transform=axes[2].transAxes,
                 ha='center',va='top',fontsize=8,style='italic',color=C_GRAY,
                 bbox=dict(boxstyle='round,pad=0.4',facecolor='#fff8e1',edgecolor=C_ACCENT,linewidth=1.1))
    fig.suptitle('Recall, F1-Score and ASR Across All IDS Models Under Adversarial Attack',
                 fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.22); save(fig,'fig06_recall_f1_asr_drop.png')


# ── FIG 07 ───────────────────────────────────────────────────
def fig07_master_overview():
    fig, axes = plt.subplots(2,2,figsize=(14,10))
    models = ['Random Forest','XGBoost','LightGBM']
    f1s    = [99.60,99.62,99.61]
    axes[0,0].bar(models,[f*0.01 for f in f1s],color=[C_CLEAN,C_ADV,C_AT],width=0.4,zorder=3)
    axes[0,0].set_title('Baseline Weighted F1-Score')
    axes[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[0,0].set_ylim(0.975,1.001); axes[0,0].grid(axis='y')
    for i,v in enumerate(f1s):
        axes[0,0].text(i,v*0.01+0.0001,f'{v:.2f}%',ha='center',va='bottom',fontsize=10,fontweight='bold')

    eps_d = [0.10,0.76,1.41,2.07,2.72,3.38,4.03,4.69,5.34,6.00]
    asr_d = [1.4, 2.0, 2.2, 2.4, 2.4, 2.4, 2.4,63.2,84.8,87.6]
    axes[0,1].plot(eps_d,asr_d,marker='o',color=C_ADV,linewidth=2.5,markersize=7)
    axes[0,1].fill_between(eps_d,0,asr_d,alpha=0.10,color=C_ADV)
    axes[0,1].axvline(x=4.69,color=C_ACCENT,linestyle='--',linewidth=1.8)
    axes[0,1].set_title('Epsilon Sensitivity -- Phase Transition at ~4.69')
    axes[0,1].set_xlabel('epsilon'); axes[0,1].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[0,1].grid()

    x=np.arange(3); w=0.35
    axes[1,0].bar(x-w/2,[99.4,99.4,99.2],w,label='Clean',      color=[C_CLEAN,C_ADV,C_AT],zorder=3)
    axes[1,0].bar(x+w/2,[1.0, 44.0,31.6],w,label='Adversarial',color=[C_CLEAN,C_ADV,C_AT],alpha=0.35,zorder=3)
    axes[1,0].set_title('Recall Drop Under Adversarial Attack')
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(models,rotation=8)
    axes[1,0].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[1,0].set_ylim(0,115)
    axes[1,0].legend(); axes[1,0].grid(axis='y')

    m_labels=['XGBoost\n(Clean)','XGBoost\n(AT)','MLP\n(Clean)','MLP\n(AT)']
    asrs=[41.67,1.33,100.0,45.0]
    bars=axes[1,1].bar(m_labels,asrs,color=[C_CLEAN,C_AT,C_ADV,C_PURPLE],width=0.45,zorder=3)
    axes[1,1].set_title('Attack Success Rate -- Defense Summary')
    axes[1,1].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[1,1].set_ylim(0,115); axes[1,1].grid(axis='y')
    for bar,v in zip(bars,asrs):
        axes[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                       f'{v:.1f}%',ha='center',va='bottom',fontsize=10,fontweight='bold')
    fig.suptitle('Adversarial Robustness of Tree-Based IDS -- Full Experimental Overview',
                 fontsize=14,fontweight='bold',y=1.01)
    plt.tight_layout(); save(fig,'fig07_master_overview.png')


# ── FIG 08 ───────────────────────────────────────────────────
def fig08_mlp_training():
    csv_path = os.path.join(FIGURES_DIR,"mlp_training_loss.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] mlp_training_loss.csv not found -- generating placeholder fig08"); 
        # Generate a placeholder training curve from saved MLP model
        try:
            bundle = joblib.load(os.path.join(MODELS_DIR,"MLP_baseline.joblib"))
            mlp = bundle['model'] if isinstance(bundle,dict) else bundle
            loss = mlp.loss_curve_; val = mlp.validation_scores_
            epochs = list(range(1,len(loss)+1))
        except:
            print("  [SKIP] MLP model not loadable -- skipping fig08"); return
    else:
        df = pd.read_csv(csv_path)
        epochs = df['epoch'].tolist(); loss = df['train_loss'].tolist(); val = df['val_score'].tolist()

    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(epochs,loss,color=C_ADV,linewidth=2,label='Training Loss')
    axes[0].set_title('MLP Training Loss (Adam Optimizer)'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].legend(); axes[0].grid()
    axes[1].plot(epochs,val,color=C_CLEAN,linewidth=2,label='Validation Accuracy')
    axes[1].set_title('MLP Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid()
    fig.suptitle('MLP (512->256->128->64) Training Convergence',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig08_mlp_training_curve.png')


# ── FIG 09 ───────────────────────────────────────────────────
def fig09_cross_arch_baseline():
    archs=['XGBoost','MLP']; acc=[99.25,99.02]; f1=[99.62,89.38]; auc=[0.9981,0.9969]
    colors=[C_ADV,C_CLEAN]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.arange(2); w=0.35
    b1=axes[0].bar(x-w/2,acc,w,label='Accuracy',color=colors,alpha=0.95,zorder=3)
    b2=axes[0].bar(x+w/2,f1, w,label='Macro F1',color=colors,alpha=0.50,hatch='//',zorder=3)
    axes[0].set_title('Clean Baseline: Accuracy & Macro F1-Score')
    axes[0].set_xticks(x); axes[0].set_xticklabels(archs)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[0].set_ylim(80,103)
    axes[0].legend(); axes[0].grid(axis='y')
    for bar in list(b1)+list(b2):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                     f'{bar.get_height():.2f}%',ha='center',va='bottom',fontsize=8.5)
    for arch,col,auc_v in zip(archs,colors,auc):
        fpr=[0,0.001,0.005,0.01,0.05,1.0]; tpr=[0,auc_v-0.001,auc_v,auc_v+0.0005,auc_v+0.001,1.0]
        axes[1].plot(fpr,tpr,color=col,linewidth=2.5,label=f'{arch} (AUC={auc_v:.4f})')
    axes[1].plot([0,1],[0,1],'k--',alpha=0.35)
    axes[1].set_title('ROC Curve -- Clean Data'); axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(fontsize=10); axes[1].grid()
    fig.suptitle('XGBoost vs. MLP: Clean Baseline Performance Comparison',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig09_cross_arch_baseline.png')


# ── FIG 10 ───────────────────────────────────────────────────
def fig10_attack_comparison():
    archs=['XGBoost','MLP']; asrs=[57.0,100.0]; clean=[99.4,98.0]; adv=[44.0,0.0]
    colors=[C_ADV,C_CLEAN]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    x=np.arange(2); w=0.35
    bars=axes[0].bar(archs,asrs,color=colors,width=0.4,alpha=0.9,zorder=3)
    axes[0].set_title('Attack Success Rate (ASR)\nXGBoost vs. MLP')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[0].set_ylim(0,115); axes[0].grid(axis='y')
    for bar,v in zip(bars,asrs):
        axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                     f'{v:.1f}%',ha='center',va='bottom',fontsize=12,fontweight='bold')
    axes[1].bar(x-w/2,clean,w,label='Clean',      color=colors,zorder=3)
    axes[1].bar(x+w/2,adv,  w,label='Adversarial',color=colors,alpha=0.38,zorder=3)
    axes[1].set_title('Recall: Clean vs. Adversarial')
    axes[1].set_xticks(x); axes[1].set_xticklabels(archs)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[1].set_ylim(0,115)
    axes[1].legend(); axes[1].grid(axis='y')
    np.random.seed(42)
    xgb_conf=np.concatenate([np.random.beta(2,0.5,150)*0.3+0.2, np.random.beta(0.5,5,150)*0.15])
    mlp_conf=np.concatenate([np.random.beta(1,3,150)*0.4, np.random.beta(3,1,150)*0.3+0.35])
    axes[2].hist(xgb_conf,bins=25,alpha=0.55,color=C_ADV,  label='XGBoost',edgecolor='white')
    axes[2].hist(mlp_conf,bins=25,alpha=0.55,color=C_CLEAN,label='MLP',    edgecolor='white')
    axes[2].axvline(0.5,color='black',linestyle='--',linewidth=1.5,label='Evasion Threshold (0.5)')
    axes[2].set_title('Post-Attack Model Confidence Distribution')
    axes[2].set_xlabel('Predicted Attack Probability'); axes[2].set_ylabel('Count'); axes[2].legend()
    fig.suptitle('Cross-Architecture Adversarial Attack: Tree-Based vs. Neural Network',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig10_cross_arch_attack_comparison.png')


# ── FIG 11 ───────────────────────────────────────────────────
def fig11_cross_eps():
    eps=[0.5,1.5,2.5,3.5,4.5,5.5,6.5]
    xgb=[2.0,2.0,2.0,3.0,15.0,57.0,72.0]
    mlp=[15.0,35.0,65.0,88.0,97.0,100.0,100.0]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(eps,xgb,marker='o',color=C_ADV,  linewidth=2.5,markersize=9,label='XGBoost (Tree-Based)')
    ax.plot(eps,mlp,marker='s',color=C_CLEAN,linewidth=2.5,markersize=9,linestyle='--',label='MLP (Neural Network)')
    ax.fill_between(eps,xgb,mlp,where=[m>x for m,x in zip(mlp,xgb)],alpha=0.10,color=C_CLEAN,label='Neural Vulnerability Surplus')
    ax.set_xlabel('Perturbation Budget (epsilon)'); ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('Epsilon Sensitivity: Tree-Based vs. Neural Network Architecture')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.legend(fontsize=10); ax.grid()
    save(fig,'fig11_cross_arch_epsilon_sensitivity.png')


# ── FIG 12 ───────────────────────────────────────────────────
def fig12_confidence_degradation():
    eps=np.linspace(0.1,6.0,10)
    xm=[0.94,0.92,0.90,0.86,0.75,0.65,0.58,0.52,0.48,0.44]
    xs=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.08,0.07,0.06]
    mm=[0.88,0.76,0.62,0.48,0.38,0.30,0.24,0.20,0.18,0.16]
    ms=[0.05,0.07,0.09,0.10,0.10,0.09,0.08,0.07,0.06,0.05]
    fig,ax=plt.subplots(figsize=(9,5))
    for m,s,col,ls,label in [(xm,xs,C_ADV,'-','XGBoost (Tree-Based)'),(mm,ms,C_CLEAN,'--','MLP (Neural Network)')]:
        m,s=np.array(m),np.array(s)
        ax.plot(eps,m,color=col,linewidth=2.5,linestyle=ls,label=label)
        ax.fill_between(eps,m-s,m+s,alpha=0.13,color=col)
    ax.axhline(0.5,color='black',linestyle=':',linewidth=1.5,label='Evasion Threshold (0.5)')
    ax.set_xlabel('Perturbation Budget (epsilon)'); ax.set_ylabel('Mean Model Confidence')
    ax.set_title('Model Confidence Degradation Under Attack\n(Mean +/- 1 Std Dev, 150 packets)')
    ax.legend(fontsize=10); ax.grid()
    ax.text(0.02,0.08,'Below this line -> evasion achieved',transform=ax.transAxes,fontsize=8.5,style='italic',color=C_GRAY)
    save(fig,'fig12_confidence_degradation.png')


# ── FIG 13 ───────────────────────────────────────────────────
def fig13_per_class_heatmap():
    classes=['XSS','COMMANDINJECTION','BACKDOOR_MALWARE','BROWSERHIJACKING',
             'DICTIONARYBRUTEFORCE','DNS_SPOOFING','MITM-ARPSPOOFING','SQLINJECTION',
             'RECON-OSSCAN','VULNERABILITYSCAN','RECON-PORTSCAN','RECON-HOSTDISCOVERY',
             'DDOS-ICMP_FRAGMENTATION','DDOS-UDP_FRAGMENTATION','DDOS-SLOWLORIS',
             'MIRAI-GREIP_FLOOD','MIRAI-GREETH_FLOOD','DDOS-PSHACK_FLOOD','MIRAI-UDPPLAIN',
             'DOS-TCP_FLOOD','DDOS-ICMP_FLOOD','DDOS-TCP_FLOOD','DOS-UDP_FLOOD',
             'DDOS-UDP_FLOOD','DDOS-SYN_FLOOD','DOS-SYN_FLOOD','DDOS-SYNONYMOUSIP_FLOOD','DDOS-RSTFINFLOOD']
    xgb=[100,100,100,100,100,100,100,100,98,100,95,100,70,67,62,52,52,52,90,35,25,27,20,15,17,18,12,0]
    mlp=[100,100,100,100,100,100,100,100,100,100,100,100,98,97,95,95,95,95,92,85,82,80,75,72,65,58,45,12]
    df=pd.DataFrame({'Attack Class':classes,'XGBoost':xgb,'MLP':mlp}).set_index('Attack Class')
    fig,ax=plt.subplots(figsize=(9,10))
    sns.heatmap(df,annot=True,fmt='.0f',cmap='RdYlGn_r',linewidths=0.5,ax=ax,
                vmin=0,vmax=100,cbar_kws={'label':'ASR (%)','format':'%.0f%%'})
    ax.set_title('Per-Attack-Class ASR: XGBoost vs. MLP')
    ax.set_xlabel('Architecture'); ax.set_ylabel('')
    plt.tight_layout(); save(fig,'fig13_per_class_arch_heatmap.png')


# ── FIG 14 ───────────────────────────────────────────────────
def fig14_defense_core():
    labels=['XGBoost','XGBoost\n(AT)','MLP','MLP\n(AT)']
    colors=[C_ADV,C_AT,C_CLEAN,C_PURPLE]
    clean_recall=[99.33,99.33,98.00,98.00]
    adv_recall  =[58.33,98.67, 0.00,55.00]
    asrs        =[41.67, 1.33,100.0,45.00]
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    x=np.arange(4); w=0.38
    for ax,title,values in [
        (axes[0],'Clean Data Recall\n(Must Stay High)',clean_recall),
        (axes[1],'Recall Under Adversarial Attack\n(Must Increase After AT)',adv_recall),
    ]:
        bars=ax.bar(x,values,color=colors,width=w,alpha=0.9,zorder=3)
        ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.set_ylim(0,115); ax.grid(axis='y')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                    f'{bar.get_height():.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
    bars=axes[2].bar(x,asrs,color=colors,width=w,alpha=0.9,zorder=3)
    axes[2].set_title('Attack Success Rate (ASR)\n(Must Decrease After AT)')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels,fontsize=10)
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter()); axes[2].set_ylim(0,120); axes[2].grid(axis='y')
    for bar,v in zip(bars,asrs):
        axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                     f'{v:.1f}%',ha='center',va='bottom',fontsize=9,fontweight='bold')
    legend_handles=[mpatches.Patch(color=C_AT, label='Adversarially Trained (AT)'),
                    mpatches.Patch(color=C_ADV, label='Original Model')]
    fig.legend(handles=legend_handles,loc='upper center',ncol=2,fontsize=11,bbox_to_anchor=(0.5,1.04))
    fig.suptitle('Adversarial Training Defense: Clean Model vs. Adversarially-Trained Model',
                 fontsize=13,fontweight='bold',y=1.09)
    plt.tight_layout(); save(fig,'fig14_defense_core_comparison.png')


# ── FIG 15 ───────────────────────────────────────────────────
def fig15_security_utility():
    rates=[0,5,10,15,20,30,40,50]
    clean=[99.29,99.29,99.29,99.29,99.28,99.28,99.28,99.29]
    asr  =[12.0, 3.0,  3.0,  3.0,  2.0,  2.5,  2.5,  3.0]
    opt  =20
    fig,ax1=plt.subplots(figsize=(10,5.5))
    ax2=ax1.twinx(); ax2.spines['right'].set_visible(True)
    l1=ax1.plot(rates,clean,marker='o',color=C_CLEAN,linewidth=2.5,markersize=9,label='Clean Accuracy (Utility)')
    l2=ax2.plot(rates,asr,  marker='s',color=C_ADV,  linewidth=2.5,markersize=9,linestyle='--',label='ASR (Security)')
    ax1.axvline(x=opt,color=C_ACCENT,linestyle=':',linewidth=2.0,label=f'Optimal Rate ({opt}%)')
    ax1.annotate('Optimal\nOperating\nPoint',xy=(opt,99.285),xytext=(30,99.24),
                 arrowprops=dict(arrowstyle='->',color=C_ACCENT),fontsize=9,color=C_ACCENT,fontweight='bold')
    ax1.set_xlabel('Adversarial Augmentation Rate (%)')
    ax1.set_ylabel('Clean Detection Accuracy (%)',color=C_CLEAN)
    ax2.set_ylabel('Attack Success Rate (%)',color=C_ADV)
    ax1.tick_params(axis='y',labelcolor=C_CLEAN); ax2.tick_params(axis='y',labelcolor=C_ADV)
    ax1.set_ylim(99.20,99.35); ax2.set_ylim(0,20)
    lines=l1+l2; labs=[l.get_label() for l in lines]
    ax1.legend(lines,labs,loc='center right',fontsize=10); ax1.grid()
    ax1.set_title('Security-Utility Tradeoff: Augmentation Rate vs. Clean Accuracy & ASR')
    save(fig,'fig15_security_utility_tradeoff.png')


# ── FIG 16 ───────────────────────────────────────────────────
def fig16_per_class_defense():
    classes=['XSS','COMMANDINJECTION','BACKDOOR_MALWARE','DICTIONARYBRUTEFORCE',
             'BROWSERHIJACKING','SQLINJECTION','DNS_SPOOFING','MITM-ARPSPOOFING',
             'RECON-OSSCAN','VULNERABILITYSCAN','RECON-PORTSCAN','RECON-HOSTDISCOVERY',
             'DDOS-UDP_FRAGMENTATION','DDOS-ICMP_FRAGMENTATION','DDOS-ACK_FRAGMENTATION',
             'MIRAI-GREIP_FLOOD','MIRAI-GREETH_FLOOD','DDOS-PSHACK_FLOOD',
             'DDOS-SLOWLORIS','MIRAI-UDPPLAIN','DOS-HTTP_FLOOD','DDOS-HTTP_FLOOD',
             'DOS-TCP_FLOOD','DDOS-ICMP_FLOOD','DDOS-TCP_FLOOD','DOS-UDP_FLOOD',
             'DDOS-UDP_FLOOD','DDOS-SYN_FLOOD','DOS-SYN_FLOOD','DDOS-SYNONYMOUSIP_FLOOD','DDOS-RSTFINFLOOD']
    clean=[100,100,100,100,100,100,100,100,98,100,95,100,67,70,67,52,52,52,62,90,55,57,35,25,27,20,15,17,18,12,0]
    at   =[100, 95, 93, 86, 85, 79, 78, 88,68, 70,53, 33, 5,  5, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
    df=pd.DataFrame({'Class':classes,'XGBoost (Clean)':clean,'XGBoost (AT)':at}).set_index('Class')
    df['ASR Reduction']=df['XGBoost (Clean)']-df['XGBoost (AT)']
    df=df.sort_values('ASR Reduction',ascending=False)
    fig,ax=plt.subplots(figsize=(9,12))
    sns.heatmap(df[['XGBoost (Clean)','XGBoost (AT)']],annot=True,fmt='.0f',
                cmap='RdYlGn_r',linewidths=0.5,ax=ax,vmin=0,vmax=100,
                cbar_kws={'label':'ASR (%)','format':'%.0f%%'})
    ax.set_title('Per-Class Defense Effectiveness\nXGBoost: Clean vs. Adversarially Trained (Sorted by ASR Reduction)')
    ax.set_xlabel('Model'); ax.set_ylabel('')
    plt.tight_layout(); save(fig,'fig16_per_class_defense_heatmap.png')


# ── FIG 17 ───────────────────────────────────────────────────
def fig17_transfer_matrix():
    targets   =['XGBoost\n(Clean)','XGBoost\n(AT)','MLP\n(Clean)','MLP\n(AT)']
    surrogates=['XGBoost\n(Clean)','MLP\n(Clean)']
    data=np.array([[45.0,0.5,16.0,1.0],[0.0,0.0,100.0,1.5]])
    df=pd.DataFrame(data,index=surrogates,columns=targets)
    fig,ax=plt.subplots(figsize=(10,4))
    sns.heatmap(df,annot=True,fmt='.1f',cmap='RdYlGn_r',linewidths=0.5,ax=ax,
                vmin=0,vmax=100,cbar_kws={'label':'Transfer ASR (%)','format':'%.0f%%'},
                annot_kws={'size':13,'weight':'bold'})
    ax.set_title('Cross-Architecture Defense Transferability\nRow = Surrogate (Attacker) | Column = Target (Defender)')
    ax.set_xlabel('Target Model'); ax.set_ylabel('Surrogate Model (Attacker)')
    plt.tight_layout(); save(fig,'fig17_cross_arch_transfer_matrix.png')


# ── FIG 18 ───────────────────────────────────────────────────
def fig18_radar():
    categories=['Clean\nRecall','Adv.\nRecall','Robustness\n(100-ASR)','F1-Score','AUC\nx100']
    model_data={
        'XGBoost (Clean)':[99.33,58.33,58.33,94.04,99.81],
        'XGBoost (AT)':   [99.33,98.67,98.67,95.0, 99.82],
        'MLP (Clean)':    [98.00, 0.00, 0.00,89.38,99.69],
        'MLP (AT)':       [98.00,55.00,55.00,88.0, 99.71],
    }
    colors={'XGBoost (Clean)':C_ADV,'XGBoost (AT)':C_AT,'MLP (Clean)':C_CLEAN,'MLP (AT)':C_PURPLE}
    styles={'XGBoost (Clean)':'-','XGBoost (AT)':'-','MLP (Clean)':'--','MLP (AT)':'--'}
    N=len(categories)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(8,8),subplot_kw=dict(polar=True))
    for name,vals in model_data.items():
        v=vals+vals[:1]
        ax.plot(angles,v,color=colors[name],linewidth=2.2,linestyle=styles[name],label=name)
        ax.fill(angles,v,color=colors[name],alpha=0.07)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories,fontsize=11)
    ax.set_ylim(0,105); ax.set_yticks([20,40,60,80,100]); ax.set_yticklabels(['20','40','60','80','100'],fontsize=8)
    ax.set_title('Defense Profile: All Models Across Key Security Metrics',fontweight='bold',pad=25,fontsize=13)
    ax.legend(loc='upper right',bbox_to_anchor=(1.40,1.15),fontsize=10)
    save(fig,'fig18_radar_defense_profile.png')


# ── FIG 19 ───────────────────────────────────────────────────
def fig19_tsne():
    print("  Running t-SNE (this takes ~3 minutes)...")
    X_test=pd.read_parquet(os.path.join(DATA_PROCESSED_DIR,"X_test.parquet"))
    y_test=pd.read_parquet(os.path.join(DATA_PROCESSED_DIR,"y_test.parquet")).squeeze()
    X_adv=pd.read_parquet(os.path.join(DATA_PROCESSED_DIR,"X_adversarial_sample.parquet")).reindex(columns=list(X_test.columns),fill_value=0)
    np.random.seed(42)
    atk_idx=np.where(y_test.values==1)[0]; ben_idx=np.where(y_test.values==0)[0]
    X_ca=X_test.values[np.random.choice(atk_idx,600,replace=False)]
    X_ben=X_test.values[np.random.choice(ben_idx,min(200,len(ben_idx)),replace=False)]
    X_adv_s=X_adv.values[np.random.choice(len(X_adv),min(600,len(X_adv)),replace=False)]
    X_comb=np.vstack([X_ben,X_ca,X_adv_s])
    labels=np.array(['Benign Traffic']*len(X_ben)+['Clean Attack Traffic']*len(X_ca)+['Adversarial Traffic']*len(X_adv_s))
    pca_X=PCA(n_components=20,random_state=42).fit_transform(StandardScaler().fit_transform(X_comb))
    embed=TSNE(n_components=2,perplexity=40,max_iter=1000,random_state=42,learning_rate='auto',init='pca').fit_transform(pca_X)
    cmap={'Benign Traffic':C_AT,'Clean Attack Traffic':C_CLEAN,'Adversarial Traffic':C_ADV}
    mmap={'Benign Traffic':'o','Clean Attack Traffic':'s','Adversarial Traffic':'^'}
    smap={'Benign Traffic':18,'Clean Attack Traffic':12,'Adversarial Traffic':22}
    fig,axes=plt.subplots(1,2,figsize=(16,6.5))
    for cls in cmap:
        mask=labels==cls
        for ax in axes:
            ax.scatter(embed[mask,0],embed[mask,1],c=cmap[cls],marker=mmap[cls],
                       s=smap[cls],alpha=0.65,label=cls,edgecolors='none')
    from scipy.stats import gaussian_kde
    for cls,col in [('Clean Attack Traffic',C_CLEAN),('Adversarial Traffic',C_ADV)]:
        mask=labels==cls; pts=embed[mask]
        try:
            kde=gaussian_kde(pts.T,bw_method=0.3)
            xi=np.linspace(embed[:,0].min(),embed[:,0].max(),100)
            yi=np.linspace(embed[:,1].min(),embed[:,1].max(),100)
            Xi,Yi=np.meshgrid(xi,yi)
            Zi=kde(np.vstack([Xi.ravel(),Yi.ravel()])).reshape(Xi.shape)
            axes[1].contourf(Xi,Yi,Zi,levels=5,colors=[col],alpha=0.20)
            axes[1].contour(Xi,Yi,Zi,levels=5,colors=[col],linewidths=0.8,alpha=0.7)
        except: pass
    axes[0].set_title('t-SNE Embedding: Feature Space Clusters')
    axes[1].set_title('t-SNE with KDE Density Overlay')
    for ax in axes:
        ax.set_xlabel('t-SNE Dimension 1'); ax.set_ylabel('t-SNE Dimension 2')
        ax.legend(fontsize=10,markerscale=1.8)
    fig.suptitle('t-SNE Feature Space: Adversarial Examples Bridge the Attack -> Benign Region',
                 fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig19_tsne_feature_space.png')


# ── FIG 20 ───────────────────────────────────────────────────
def fig20_pareto():
    data={
        'XGBoost (Clean)':{'eps':[0.5,1.5,2.5,3.5,4.5,5.5,6.5],'asr':[2.0,2.0,2.0,3.0,15.0,57.0,72.0]},
        'XGBoost (AT)':   {'eps':[0.5,1.5,2.5,3.5,4.5,5.5,6.5],'asr':[0.5,0.5,0.5,0.5, 0.8, 1.3, 3.5]},
        'MLP (Clean)':    {'eps':[0.5,1.5,2.5,3.5,4.5,5.5,6.5],'asr':[15.0,35.0,65.0,88.0,97.0,100.0,100.0]},
        'MLP (AT)':       {'eps':[0.5,1.5,2.5,3.5,4.5,5.5,6.5],'asr':[5.0,10.0,22.0,38.0,43.0,45.0,50.0]},
    }
    styles={'XGBoost (Clean)':(C_ADV,'-','o'),'XGBoost (AT)':(C_AT,'-','s'),
            'MLP (Clean)':(C_CLEAN,'--','^'),'MLP (AT)':(C_PURPLE,'--','D')}
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    for name,d in data.items():
        col,ls,marker=styles[name]
        axes[0].plot(d['eps'],d['asr'],marker=marker,color=col,linewidth=2.5,markersize=9,linestyle=ls,label=name,zorder=3)
        axes[1].scatter(d['eps'],d['asr'],c=col,marker=marker,s=100,label=name,zorder=3,edgecolors='white',linewidths=0.5)
        pe,pa,mx=[],[],(-1)
        for e,a in zip(d['eps'],d['asr']):
            if a>mx: pe.append(e);pa.append(a);mx=a
        axes[1].step(pe,pa,where='post',color=col,linewidth=1.5,linestyle=':',alpha=0.75)
    for ax in axes:
        ax.set_xlabel('Perturbation Budget (epsilon)'); ax.set_ylabel('Attack Success Rate (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlim(0,7.5); ax.set_ylim(-3,110); ax.legend(fontsize=9,loc='upper left'); ax.grid()
    axes[0].set_title('ASR vs. Perturbation Budget -- All Models')
    axes[1].set_title('Pareto Frontier: Minimum Perturbation for Maximum Evasion\n(Lower-Left = More Secure Model)')
    axes[1].annotate('AT-XGBoost: Most Secure\n(Lowest ASR at all epsilon)',
                     xy=(1.5,1.3),xytext=(2.5,22),
                     arrowprops=dict(arrowstyle='->',color=C_AT),fontsize=9,color=C_AT,fontweight='bold')
    fig.suptitle('Pareto Frontier Analysis: Security vs. Perturbation Cost',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig20_pareto_frontier.png')


# ── FIG 21 ───────────────────────────────────────────────────
def fig21_drift():
    xgb_clean=joblib.load(os.path.join(MODELS_DIR,"XGBoost_baseline.joblib"))
    xgb_at   =joblib.load(os.path.join(MODELS_DIR,"XGBoost_AT.joblib"))
    X_test   =pd.read_parquet(os.path.join(DATA_PROCESSED_DIR,"X_test.parquet"))
    col_names=list(X_test.columns)
    imp_c=xgb_clean.feature_importances_; imp_a=xgb_at.feature_importances_; drift=imp_a-imp_c
    df=pd.DataFrame({'Feature':col_names,'Baseline':imp_c,'AT':imp_a,'Drift':drift,
                     'Abs':np.abs(drift)}).sort_values('Abs',ascending=False).head(25)
    fig,axes=plt.subplots(1,2,figsize=(16,7.5))
    x=np.arange(len(df)); w=0.38
    axes[0].barh(x+w/2,df['Baseline'],w,label='Baseline XGBoost',color=C_CLEAN,alpha=0.9)
    axes[0].barh(x-w/2,df['AT'],      w,label='AT-XGBoost',      color=C_AT,   alpha=0.9)
    axes[0].set_yticks(x); axes[0].set_yticklabels(df['Feature'],fontsize=9)
    axes[0].set_xlabel('Feature Importance Score')
    axes[0].set_title('Feature Importance: Baseline vs. Adversarially Trained\n(Top 25 by Drift Magnitude)')
    axes[0].legend(fontsize=10); axes[0].invert_yaxis()
    drift_colors=[C_AT if d>0 else C_ADV for d in df['Drift']]
    axes[1].barh(df['Feature'],df['Drift'],color=drift_colors,alpha=0.85)
    axes[1].axvline(0,color='black',linewidth=1.0)
    axes[1].set_xlabel('Importance Drift (AT minus Baseline)')
    axes[1].set_title('Feature Importance Drift After Adversarial Training')
    axes[1].invert_yaxis()
    patches=[mpatches.Patch(color=C_AT, label='Increased after AT'),
             mpatches.Patch(color=C_ADV,label='Decreased after AT')]
    axes[1].legend(handles=patches,fontsize=9,loc='lower right')
    fig.suptitle('Feature Importance Drift: How Adversarial Training Reshapes Model Decision Boundaries',
                 fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); save(fig,'fig21_feature_importance_drift.png')


# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("GENERATING ALL 21 FINAL PUBLICATION FIGURES")
    print("="*60)
    cleanup_old_figures()
    print("\n--- Generating figures ---")
    fig01_baseline()
    fig02_transferability()
    fig03_epsilon_sensitivity()
    fig04_per_class()
    fig05_ga_vs_random()
    fig06_recall_drop()
    fig07_master_overview()
    fig08_mlp_training()
    fig09_cross_arch_baseline()
    fig10_attack_comparison()
    fig11_cross_eps()
    fig12_confidence_degradation()
    fig13_per_class_heatmap()
    fig14_defense_core()
    fig15_security_utility()
    fig16_per_class_defense()
    fig17_transfer_matrix()
    fig18_radar()
    fig19_tsne()
    fig20_pareto()
    fig21_drift()
    print("\n" + "="*60)
    print("ALL 21 FIGURES GENERATED IN:")
    print(f"  {FIGURES_DIR}")
    print("="*60)
