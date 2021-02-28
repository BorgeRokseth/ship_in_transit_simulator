from math import log, e
import scipy.stats as st

' 1 represent towing, 2 represent disconnecting.'


def action_cost(action):
    if action == 1:
        C_action = -2
    elif action == 2:
        C_action = -10
    return C_action


def action_success_rate_log(action, t):
    if action == 2:
        p_action = 1
    else:
        if t >= 11:
            p_action = 1
        elif t < 4:
            p_action = 0
        else:
            p_action = log(t - 3, 9)
    return p_action


def action_success_rate_normal(action, t, mu, sigma):
    if action == 1:
        p_action = st.norm.cdf(t, mu, sigma)
    elif action == 2:
        p_action = 1
    return p_action


def tpr(t, a_tpr, b_tpr):
    TPR = a_tpr + b_tpr * log(13 - t, 12)
    return TPR


def fpr(t, a_fpr, b_fpr):
    FPR = a_fpr + b_fpr * log(13 - t, 12)
    return FPR


def voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, t, action):
    TPR = a_tpr + b_tpr * log(13 - t, 12)

    FPR = a_fpr + b_fpr * log(13 - t, 12)
    p_action = action_success_rate_log(action, t)
    C_action = action_cost(action)

    VoI_precautionary = (1 - p_accident) * C_noaccident + C_action + (1 - p_action) * C_accident
    VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
    max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
    V = ((1 - p_accident) * C_noaccident) + (p_accident * C_accident) + (
                (1 - p_accident) * FPR * (C_action - C_noaccident)) + (
                    p_accident * TPR * (C_action - p_action * C_accident))
    VoP = V - max_VoI
    return TPR, FPR, max_VoI, V, VoP


def voi_2(p_accident, C_accident, p_action, TPR, FPR, C_noaccident, action):
    C_action = action_cost(action)
    VoI_precautionary = (1 - p_accident) * C_noaccident + C_action + (1 - p_action) * C_accident
    VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
    max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
    V = ((1 - p_accident) * C_noaccident) + (p_accident * C_accident) + (
                (1 - p_accident) * FPR * (C_action - C_noaccident)) + (
                    p_accident * TPR * (C_action - p_action * C_accident))
    VoP = V - max_VoI
    return max_VoI, V, VoP


t = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
p_accident = 0.1
C_accident = -100
p_action = [1, 1, 1, 1, 0.95, 0.8, 0.7, 0.5, 0, 0, 0, 0]
# p_action = [1,1,1,1,1,1,1,1,1,1,1,1]
action = 1
a_tpr = 0.5
b_tpr = 0.48
a_fpr = 0.1556
b_fpr = -0.1445
C_noaccident = 0

'''For discrete calcaulation'''


def get_optimal_time():
    max_V = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]
    optimal_time = t[0]

    for i in t:
        VoI_t = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
        print(action_success_rate_log(2, i))
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

def time_range(start, end, step):
    while start <= end:
        yield start
        start += step
def get_optimal_time_continuous(action):
    max_V = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]

    for i in time_range(1, 12, 0.1):
        VoI_t = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
        # print(i)
        # print(action_success_rate_normal(1,i))
        if VoI_t[3] >= max_V:
            max_V = VoI_t[3]
            optimal_time = i
    return max_V, optimal_time


#print(get_optimal_time())
#print(get_optimal_time_continuous(1)[1])


def sensitivity_study_accident_cost():
    optimal_time_list = []
    optimal_value_list = []
    optimal_action_list = []

    for k in my_range(-37, -3, 1):
        max_V = [None] * 2
        VoI_precautionary = [None] * 2
        time = [None] * 2
        list = []
        C_accident = k

        VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident

        for j in [1, 2]:
            action = j
            max_V[j - 1] = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]

            for i in time_range(1, 12, 0.1):
                VoI_t = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
                # print(i)
                # print(action_success_rate_normal(1,i))
                if VoI_t[3] >= max_V[j - 1]:
                    max_V[j - 1] = VoI_t[3]
                    time[j - 1] = i

            VoI_precautionary[j - 1] = action_cost(j)
        list.extend(max_V)
        list.extend(VoI_precautionary)
        list.append(VoI_neglecting_risk)
        optimal_value = max(list)
        max_index = list.index(optimal_value)
        if max_index <= 1:
            optimal_action = max_index + 1
            optimal_time = time[max_index]
        elif max_index == 4:
            optimal_action = 0
            optimal_time = 0
        else:
            optimal_action = "being precautionary and take action " + str(max_index - 1)
            optimal_time = 12

        optimal_time_list.append(optimal_time)
        optimal_value_list.append(optimal_value)
        optimal_action_list.append(optimal_action)

    return optimal_time_list, optimal_action_list, optimal_value_list


print(sensitivity_study_accident_cost())
def sensitivity_study_accident_probability():
    optimal_time_list = []
    optimal_value_list = []
    optimal_action_list = []
    C_accident = -30
    a_tpr = 0.5
    b_tpr = 0.48
    a_fpr = 0.1556
    b_fpr = -0.1445
    C_noaccident = 0
    mu = 6
    sigma = 1


    for k in my_range(0, 0.5, 0.01):
        max_V = [None] * 2
        VoI_precautionary = [None] * 2
        time = [None] * 2
        list = []
        p_accident = k

        VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident

        for j in [1, 2]:
            action = j
            max_V[j - 1] = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, 12, action)[3]

            for i in time_range(1, 12, 0.1):
                VoI_t = voi(p_accident, C_accident, a_tpr, b_tpr, a_fpr, b_fpr, C_noaccident, i, action)
                # print(i)
                # print(action_success_rate_normal(1,i))
                if VoI_t[3] >= max_V[j - 1]:
                    max_V[j - 1] = VoI_t[3]
                    time[j - 1] = i
            VoI_precautionary[j - 1] = action_cost(j)
        list.extend(max_V)
        list.extend(VoI_precautionary)
        list.append(VoI_neglecting_risk)
        optimal_value = max(list)
        max_index = list.index(optimal_value)
        if max_index <= 1:
            optimal_action = max_index + 1
            optimal_time = time[max_index]
        elif max_index == 4:
            optimal_action = 0
            optimal_time = 0
        else:
            optimal_action = "being precautionary and take action " + str(max_index - 1)
            optimal_time = 12

        optimal_time_list.append(optimal_time)
        optimal_value_list.append(optimal_value)
        optimal_action_list.append(optimal_action)

    return optimal_time_list, optimal_action_list, optimal_value_list


print(sensitivity_study_accident_probability())


def sensitivity_study_success_rate():
    optimal_time_list = []
    optimal_value_list = []
    optimal_action_list = []
    C_accident = -28
    a_tpr = 0.5
    b_tpr = 0.48
    a_fpr = 0.1556
    b_fpr = -0.1445
    C_noaccident = 0
    p_accident = 0.1
    mu = 6


    for k in my_range(0.01, 3, 0.5):
        max_V = [None] * 2
        VoI_precautionary = [None] * 2
        time = [None] * 2
        list = []
        sigma = k

        VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident

        for j in [1, 2]:
            action = j
            max_V[j - 1] = C_accident

            for i in time_range(1, 12, 0.1):
                TPR = tpr(i, a_tpr, b_tpr)
                FPR = fpr(i, a_fpr, b_fpr)
                p_action = action_success_rate_normal(j, i, mu, sigma)
                VoI_t = voi_2(p_accident, C_accident, p_action, TPR, FPR, C_noaccident, action)
                # print(i)
                # print(action_success_rate_normal(1,i))
                if VoI_t[1] >= max_V[j - 1]:
                    max_V[j - 1] = VoI_t[1]
                    time[j - 1] = i
            VoI_precautionary[j - 1] = action_cost(j)
        list.extend(max_V)
        list.extend(VoI_precautionary)
        list.append(VoI_neglecting_risk)
        optimal_value = max(list)
        max_index = list.index(optimal_value)
        if max_index <= 1:
            optimal_action = max_index + 1
            optimal_time = time[max_index]
        elif max_index == 4:
            optimal_action = 0
            optimal_time = 0
        else:
            optimal_action = "being precautionary and take action " + str(max_index - 1)
            optimal_time = 12

        optimal_time_list.append(optimal_time)
        optimal_value_list.append(optimal_value)
        optimal_action_list.append(optimal_action)

    return optimal_time_list, optimal_action_list, optimal_value_list
#print(sensitivity_study_success_rate())


def sensitivity_study_cost_ratio():
    optimal_time_list = []
    optimal_value_list = []
    optimal_action_list = []
    a_tpr = 0.5
    b_tpr = 0.48
    a_fpr = 0.1556
    b_fpr = -0.1445
    C_noaccident = 0
    p_accident = 0.1
    mu = 6
    sigma = 2
    C_action=-2
    for k in my_range(1, 50, 0.5):

        list = []
        C_accident = C_action*k

        VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
        max_V = C_accident

        for i in time_range(1, 12, 0.1):
            TPR = tpr(i, a_tpr, b_tpr)
            FPR = fpr(i, a_fpr, b_fpr)
            p_action = action_success_rate_normal(1, i, mu, sigma)
            VoI_precautionary = (1 - p_accident) * C_noaccident + C_action + (1 - p_action) * C_accident
            VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
            max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
            V = ((1 - p_accident) * C_noaccident) + (p_accident * C_accident) + (
                    (1 - p_accident) * FPR * (C_action - C_noaccident)) + (
                        p_accident * TPR * (C_action - p_action * C_accident))
            VoP = V - max_VoI
            # print(i)
            # print(action_success_rate_normal(1,i))
            if V >= VoI_neglecting_risk:
                max_V = V
                time = i
        VoI_precautionary = C_action
        list.append(max_V)
        list.append(VoI_precautionary)
        list.append(VoI_neglecting_risk)
        optimal_value = max(list)
        max_index = list.index(optimal_value)
        if max_index <= 0:
            optimal_action = max_index + 1
            optimal_time = time
        elif max_index == 2:
            optimal_action = "neglecting risk "
            optimal_time = 0
        else:
            optimal_action = "being precautionary and take action immediately"
            optimal_time = 12

        optimal_time_list.append(optimal_time)
        optimal_value_list.append(optimal_value)
        optimal_action_list.append(optimal_action)

    return optimal_time_list, optimal_action_list, optimal_value_list

#print(sensitivity_study_cost_ratio())
def sensitivity_study():
    optimal_time_list = []
    optimal_value_list = []
    optimal_action_list = []
    C_accident = -28
    a_tpr = 0.3
    b_tpr = 0.68
    a_fpr = 0.1556
    b_fpr = -0.1445
    C_noaccident = 0
    p_accident = 0.1
    mu = 5
    sigma = 2
    C_action = -2
    for k in my_range(0.01, 1, 0.01):

        list = []
        VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
        max_V = C_accident
        C_action=C_accident*k

        for i in time_range(1, 12, 0.1):
            TPR = a_tpr + b_tpr * log(13 - i, 12)
            FPR = a_fpr + b_fpr * log(13 - i, 12)
            p_action = action_success_rate_normal(1, i, mu, sigma)
            VoI_precautionary = (1 - p_accident) * C_noaccident + C_action + (1 - p_action) * C_accident
            VoI_neglecting_risk = (1 - p_accident) * C_noaccident + p_accident * C_accident
            max_VoI = max(VoI_precautionary, VoI_neglecting_risk)
            V = ((1 - p_accident) * C_noaccident) + (p_accident * C_accident) + (
                    (1 - p_accident) * FPR * (C_action - C_noaccident)) + (
                        p_accident * TPR * (C_action - p_action * C_accident))
            VoP = V - max_VoI
            # print(i)
            # print(action_success_rate_normal(1,i))
            if V >= VoI_neglecting_risk:
                max_V = V
                action_time = i
        VoI_precautionary = C_action
        list.append(max_V)
        list.append(VoI_precautionary)
        list.append(VoI_neglecting_risk)
        optimal_value = max(list)
        max_index = list.index(optimal_value)
        if max_index <= 0:
            optimal_action = max_index + 1
            optimal_time = action_time
        elif max_index == 2:
            optimal_action = "neglecting risk "
            optimal_time = 0
        else:
            optimal_action = "being precautionary and take action immediately"
            optimal_time = 12

        optimal_time_list.append(optimal_time)
        optimal_value_list.append(optimal_value)
        optimal_action_list.append(optimal_action)

    return optimal_time_list, optimal_action_list, optimal_value_list
print(sensitivity_study())