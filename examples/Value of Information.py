
def VoI(self, p_accident, C_accident, p_action, C_action, FPR, TPR, C_noaccident):
    self.VoI_precautionary = (1-p_accident)*C_noaccident+C_action+(1-p_action)*C_accident
    self.VoI_neglecting_risk = (1-p_accident)*C_noaccident+p_accident*C_accident
    self.max_VoI = max(self.VoI_precautionary, self.VoI_neglecting_risk)
    self.V = (1-p_accident)*C_noaccident+p_accident*C_accident+(1-p_accident)*FPR(C_action-C_noaccident)+p_accident*TPR*(C_action-p_action*C_accident)
    self.VoP = self.V-self.max_VoI

