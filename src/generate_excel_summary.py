"""
generate_excel_summary.py
Generates research_results_summary.xlsx with all experimental results across 9 sheets.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from config import FIGURES_DIR

OUT = os.path.join(os.path.dirname(FIGURES_DIR), "research_results_summary.xlsx")

def style_sheet(ws):
    """Apply header styling to a worksheet."""
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True, size=11)
    border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', wrap_text=True)
        cell.border = border
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(horizontal='center')
    for col in ws.columns:
        max_len = max((len(str(c.value or '')) for c in col), default=10)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 30)
    ws.freeze_panes = 'A2'


writer = pd.ExcelWriter(OUT, engine='openpyxl')

# ── Sheet 1: Baseline Models ──────────────────────────────
df = pd.DataFrame({
    'Model':            ['Random Forest', 'XGBoost', 'LightGBM'],
    'Accuracy (%)':     [99.22, 99.25, 99.24],
    'Macro F1 (%)':     [99.60, 99.62, 99.61],
    'Weighted F1 (%)':  [99.60, 99.62, 99.61],
    'AUC':              [0.9981, 0.9981, 0.9980],
    'Train Time (s)':   [10.18, 3.41, 1.99],
})
df.to_excel(writer, sheet_name='Phase1_Baseline', index=False)

# ── Sheet 2: Phase 2 Attack Results ──────────────────────
df = pd.DataFrame({
    'Model':              ['Random Forest', 'XGBoost', 'LightGBM'],
    'Clean Recall (%)':   [99.4, 99.4, 99.2],
    'Adv Recall (%)':     [1.0, 44.0, 31.6],
    'Recall Drop (pp)':   [98.4, 55.4, 67.6],
    'Clean F1 (%)':       [93.16, 94.04, 93.85],
    'Adv F1 (%)':         [1.74, 56.41, 43.95],
    'F1 Drop (pp)':       [91.42, 37.63, 49.90],
    'ASR (%)':            [99.0, 56.0, 68.4],
    'Attack Budget (e)':  [5.0, 5.0, 5.0],
    'GA Pop / Gen':       ['300 / 100', '300 / 100', '300 / 100'],
})
df.to_excel(writer, sheet_name='Phase2_Attack', index=False)

# ── Sheet 3: Epsilon Ablation ─────────────────────────────
df = pd.DataFrame({
    'Epsilon (e)':   [0.10, 0.76, 1.41, 2.07, 2.72, 3.38, 4.03, 4.69, 5.34, 6.00],
    'ASR (%)':       [1.4,  2.0,  2.2,  2.4,  2.4,  2.4,  2.4, 63.2, 84.8, 87.6],
    'Adv Examples':  [15, 18, 19, 19, 19, 19, 19, 275, 290, 295],
    'Note':          ['','','','','','','','** PHASE TRANSITION **','',''],
})
df.to_excel(writer, sheet_name='Phase2_EpsilonAblation', index=False)

# ── Sheet 4: GA vs Random ─────────────────────────────────
df = pd.DataFrame({
    'Epsilon (e)':       [0.5, 1.4, 2.3, 3.2, 4.1, 5.0],
    'GA ASR (%)':        [2.0, 2.2, 2.4, 2.4, 2.4, 72.8],
    'Random ASR (%)':    [0.2, 0.2, 0.2, 0.6, 0.8, 2.0],
    'GA Advantage (pp)': [1.8, 2.0, 2.2, 1.8, 1.6, 70.8],
})
df.to_excel(writer, sheet_name='Phase2_GA_vs_Random', index=False)

# ── Sheet 5: Per-Class ASR (Phase 2) ─────────────────────
df = pd.DataFrame({
    'Attack Class': ['DDOS-RSTFINFLOOD','DDOS-TCP_FLOOD','DDOS-SYN_FLOOD','DDOS-PSHACK_FLOOD',
                     'DOS-UDP_FLOOD','DDOS-UDP_FLOOD','MIRAI-UDPPLAIN','DDOS-SYNONYMOUSIP_FLOOD',
                     'DDOS-ICMP_FLOOD','DOS-TCP_FLOOD','DDOS-UDP_FRAGMENTATION','DOS-SYN_FLOOD',
                     'MIRAI-GREETH_FLOOD','MIRAI-GREIP_FLOOD','MITM-ARPSPOOFING','RECON-PORTSCAN',
                     'VULNERABILITYSCAN','DDOS-ICMP_FRAGMENTATION','DDOS-ACK_FRAGMENTATION',
                     'DNS_SPOOFING','RECON-HOSTDISCOVERY','DOS-HTTP_FLOOD','DICTIONARYBRUTEFORCE',
                     'RECON-OSSCAN','XSS','DDOS-SLOWLORIS','BACKDOOR_MALWARE','DDOS-HTTP_FLOOD',
                     'COMMANDINJECTION','BROWSERHIJACKING','SQLINJECTION'],
    'XGBoost ASR (%)': [0,27,17,52,20,15,30,12,25,35,67,18,52,52,100,95,100,70,67,100,100,55,100,98,100,62,100,57,100,100,100],
    'Category': ['DDoS','DDoS','DDoS','DDoS','DoS','DDoS','Mirai','DDoS','DDoS','DoS','DDoS',
                 'DoS','Mirai','Mirai','MITM','Recon','Recon','DDoS','DDoS','Spoofing',
                 'Recon','DoS','BruteForce','Recon','WebAtk','DDoS','Malware','DDoS','Injection','Hijack','Injection'],
})
df.to_excel(writer, sheet_name='Phase2_PerClass_ASR', index=False)

# ── Sheet 6: Cross-Architecture (Phase 3) ─────────────────
df = pd.DataFrame({
    'Model':               ['XGBoost', 'MLP'],
    'Architecture':        ['Gradient Boosted Trees', 'Neural Network (512-256-128-64)'],
    'Clean Accuracy (%)':  [99.25, 99.02],
    'Clean Macro F1 (%)':  [99.62, 89.38],
    'AUC':                 [0.9981, 0.9969],
    'Decision Threshold':  [0.5000, 0.9722],
    'Threshold Method':    ['Default', "Youden's J"],
    'ASR (%)':             [57.0, 100.0],
    'Clean Recall (%)':    [99.4, 98.0],
    'Adv Recall (%)':      [44.0, 0.0],
})
df.to_excel(writer, sheet_name='Phase3_CrossArch', index=False)

# ── Sheet 7: Transfer Matrix (Phase 3 + 4) ───────────────
df = pd.DataFrame({
    'Surrogate (Attacker)': ['XGBoost (Clean)', 'MLP (Clean)'],
    'XGBoost Clean (%)':    [45.0, 0.0],
    'XGBoost AT (%)':       [0.5, 0.0],
    'MLP Clean (%)':        [16.0, 100.0],
    'MLP AT (%)':           [1.0, 1.5],
})
df.to_excel(writer, sheet_name='Phase4_TransferMatrix', index=False)

# ── Sheet 8: Defense Summary (Phase 4) ───────────────────
df = pd.DataFrame({
    'Model':               ['XGBoost (baseline)', 'XGBoost (AT)', 'MLP (baseline)', 'MLP (AT)'],
    'Defense Applied':     ['None', 'Gaussian Subspace Smoothing AT', 'None', 'Gaussian Subspace Smoothing AT'],
    'Clean Recall (%)':    [99.33, 99.33, 98.00, 98.00],
    'Adv Recall (%)':      [58.33, 98.67, 0.00, 55.00],
    'Recall Drop (pp)':    [41.00, 0.67, 98.00, 43.00],
    'ASR (%)':             [41.67, 1.33, 100.00, 45.00],
    'ASR Reduction (pp)':  ['-', 40.34, '-', 55.00],
    'Pareto Dominant':     ['No', 'YES', 'No', 'No'],
})
df.to_excel(writer, sheet_name='Phase4_Defense', index=False)

# ── Sheet 9: Security-Utility Tradeoff ───────────────────
df = pd.DataFrame({
    'Aug. Rate (%)':      [0, 5, 10, 15, 20, 30, 40, 50],
    'Clean Accuracy (%)': [99.29, 99.29, 99.29, 99.29, 99.28, 99.28, 99.28, 99.29],
    'ASR (%)':            [12.0, 3.0, 3.0, 3.0, 2.0, 2.5, 2.5, 3.0],
    'Score (Acc - ASR)':  [87.29, 96.29, 96.29, 96.29, 97.28, 96.78, 96.78, 96.29],
    'Optimal':            ['No','No','No','No','YES','No','No','No'],
})
df.to_excel(writer, sheet_name='Phase4_SecurityUtility', index=False)

# ── Sheet 10: Per-Class Defense ───────────────────────────
df = pd.DataFrame({
    'Attack Class': ['XSS','COMMANDINJECTION','BACKDOOR_MALWARE','DICTIONARYBRUTEFORCE',
                     'BROWSERHIJACKING','SQLINJECTION','DNS_SPOOFING','MITM-ARPSPOOFING',
                     'RECON-OSSCAN','VULNERABILITYSCAN','RECON-PORTSCAN','RECON-HOSTDISCOVERY',
                     'DDOS-UDP_FRAGMENTATION','DDOS-ICMP_FRAGMENTATION','DDOS-ACK_FRAGMENTATION',
                     'MIRAI-GREIP_FLOOD','MIRAI-GREETH_FLOOD','DDOS-PSHACK_FLOOD',
                     'DDOS-SLOWLORIS','MIRAI-UDPPLAIN','DOS-HTTP_FLOOD','DDOS-HTTP_FLOOD',
                     'DOS-TCP_FLOOD','DDOS-ICMP_FLOOD','DDOS-TCP_FLOOD','DOS-UDP_FLOOD',
                     'DDOS-UDP_FLOOD','DDOS-SYN_FLOOD','DOS-SYN_FLOOD','DDOS-SYNONYMOUSIP_FLOOD','DDOS-RSTFINFLOOD'],
    'XGBoost ASR (%)':    [100,100,100,100,100,100,100,100,98,100,95,100,67,70,67,52,52,52,62,30,55,57,35,25,27,20,15,17,18,12,0],
    'XGBoost-AT ASR (%)': [100, 95, 93, 86, 85, 79, 78, 88, 68, 70, 53, 33,  5, 5,  0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
    'ASR Reduction (pp)': [0,5,7,14,15,21,22,12,30,30,42,67,62,65,67,52,52,52,60,28,55,57,35,25,27,20,15,17,18,12,0],
})
df.to_excel(writer, sheet_name='Phase4_PerClassDefense', index=False)

writer.close()

# Style all sheets
from openpyxl import load_workbook
wb = load_workbook(OUT)
for ws in wb.worksheets:
    style_sheet(ws)
wb.save(OUT)

print(f"[SAVED] {OUT}")
print(f"  Sheets: {', '.join(wb.sheetnames)}")
