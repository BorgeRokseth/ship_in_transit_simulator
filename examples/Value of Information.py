from math import log
def voi(p_accident, C_accident, p_action, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, t):

    TPR = a_tpr + b_tpr * log(13-t,12)

    FPR = a_fpr + b_fpr * log(13-t,12)
    VoI_precautionary = (1-p_accident) * C_noaccident + C_action + (1-p_action) * C_accident
    VoI_neglecting_risk = (1-p_accident) * C_noaccident + p_accident * C_accident
    max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
    V = (1-p_accident) * C_noaccident + p_accident * C_accident+(1 - p_accident) * FPR * (C_action - C_noaccident) + p_accident * TPR * (C_action - p_action * C_accident)
    VoP = V-max_VoI
    return max_VoI, V, VoP

t = [12,11,10,9,8,7,6,5,4,3,2,1]
p_accident = 0.1
C_accident = -1000
p_action = [1,1,1,1,0.9,0.8,0.7,0.5,0,0,0,0]
C_action = -2
a_tpr = 0.5
b_tpr = 0.48
a_fpr = 0.1556
b_fpr = -0.1445
C_noaccident = 0
for i in t:
    VoI_t = voi(p_accident, C_accident, p_action[i-12], C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i)
    print(i)
    print(VoI_t)