from math import log

' 1 represent towing, 2 represent disconnecting.'
def action_success_rate(action, t):
    if action == 2:
        p_action = 1
    else:
        if t >= 9:
            p_action=1
        elif t <4:
            p_action =0
        else:
            p_action= log(t-3, 6)
    return p_action

def voi(p_accident, C_accident, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, t, action):

    TPR = a_tpr + b_tpr * log(13-t,12)

    FPR = a_fpr + b_fpr * log(13-t,12)
    p_action=action_success_rate(action, t)


    VoI_precautionary = (1-p_accident) * C_noaccident + C_action + (1-p_action) * C_accident
    VoI_neglecting_risk = (1-p_accident) * C_noaccident + p_accident * C_accident
    max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
    V = ((1-p_accident) * C_noaccident) + (p_accident * C_accident) + ((1 - p_accident) * FPR * (C_action - C_noaccident)) + (p_accident * TPR * (C_action - p_action * C_accident))
    VoP = V-max_VoI
    return TPR, FPR, max_VoI, V, VoP

t = [12,11,10,9,8,7,6,5,4,3,2,1]
p_accident = 0.1
C_accident = -4
p_action = [1,1,1,1,0.95,0.8,0.7,0.5,0,0,0,0]
#p_action = [1,1,1,1,1,1,1,1,1,1,1,1]
C_action = -2
a_tpr = 0.5
b_tpr = 0.48
a_fpr = 0.1556
b_fpr = -0.1445
C_noaccident = 0
action = 1

'''For discrete calcaulation'''
def get_optimal_time():
    max_V = voi(p_accident, C_accident, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]
    optimal_time = t[0]

    for i in t:
        VoI_t = voi(p_accident, C_accident, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
        print(action_success_rate(1,i))
        if VoI_t[3] >= max_V:
            max_V = VoI_t[3]
            optimal_time = i
    return max_V, optimal_time

#print(get_optimal_time())
''' for continuous calculation'''



def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def get_optimal_time_continuous():
    max_V = voi(p_accident, C_accident, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]


    for i in my_range(1,12,0.1):
        VoI_t = voi(p_accident, C_accident, C_action, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
        print(action_success_rate(1,i))
        if VoI_t[3] >= max_V:
            max_V = VoI_t[3]
            optimal_time = i
    return max_V, optimal_time

print(get_optimal_time_continuous())
