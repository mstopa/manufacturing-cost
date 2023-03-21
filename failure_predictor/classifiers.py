from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEAT_NAMES = ['L0_S0_F0', 'L0_S0_F2', 'L0_S0_F4', 'L0_S0_F8', 'L0_S0_F10', 'L0_S0_F12', 'L0_S0_F14', 'L0_S0_F16', 'L0_S0_F18', 'L0_S0_F20', 'L0_S1_F24', 'L0_S1_F28', 'L0_S2_F32', 'L0_S2_F36', 'L0_S2_F40', 'L0_S2_F44', 'L0_S2_F48', 'L0_S2_F52', 'L0_S2_F56', 'L0_S2_F60', 'L0_S2_F64', 'L0_S3_F68', 'L0_S3_F72', 'L0_S3_F76', 'L0_S3_F80', 'L0_S3_F84', 'L0_S3_F88', 'L0_S3_F92', 'L0_S3_F96', 'L0_S3_F100', 'L0_S4_F104', 'L0_S4_F109', 'L0_S5_F114', 'L0_S5_F116', 'L0_S6_F118', 'L0_S6_F122', 'L0_S6_F132', 'L0_S7_F136', 'L0_S7_F138', 'L0_S7_F142', 'L0_S8_F144', 'L0_S8_F146', 'L0_S8_F149', 'L2_S26_F3036', 'L2_S26_F3040', 'L2_S26_F3047', 'L2_S26_F3051', 'L2_S26_F3055', 'L2_S26_F3062', 'L2_S26_F3069', 'L2_S26_F3073', 'L2_S26_F3077', 'L2_S26_F3106', 'L2_S26_F3113', 'L2_S26_F3117', 'L2_S26_F3121', 'L2_S26_F3125', 'L3_S29_F3315', 'L3_S29_F3318', 'L3_S29_F3321', 'L3_S29_F3324', 'L3_S29_F3327', 'L3_S29_F3333', 'L3_S29_F3336', 'L3_S29_F3339', 'L3_S29_F3345', 'L3_S29_F3348', 'L3_S29_F3351', 'L3_S29_F3354', 'L3_S29_F3357', 'L3_S29_F3360', 'L3_S29_F3373', 'L3_S29_F3379', 'L3_S29_F3407', 'L3_S29_F3427', 'L3_S29_F3433', 'L3_S29_F3461', 'L3_S29_F3464', 'L3_S29_F3473', 'L3_S29_F3479', 'L3_S29_F3482', 'L3_S29_F3485', 'L3_S29_F3488', 'L3_S29_F3491', 'L3_S30_F3494', 'L3_S30_F3504', 'L3_S30_F3509', 'L3_S30_F3514', 'L3_S30_F3519', 'L3_S30_F3524', 'L3_S30_F3534', 'L3_S30_F3544', 'L3_S30_F3554', 'L3_S30_F3564', 'L3_S30_F3569', 'L3_S30_F3574', 'L3_S30_F3579', 'L3_S30_F3584', 'L3_S30_F3589', 'L3_S30_F3594', 'L3_S30_F3599', 'L3_S30_F3604', 'L3_S30_F3609', 'L3_S30_F3614', 'L3_S30_F3619', 'L3_S30_F3624', 'L3_S30_F3629', 'L3_S30_F3634', 'L3_S30_F3639', 'L3_S30_F3654', 'L3_S30_F3659', 'L3_S30_F3664', 'L3_S30_F3669', 'L3_S30_F3674', 'L3_S30_F3684', 'L3_S30_F3694', 'L3_S30_F3699', 'L3_S30_F3704', 'L3_S30_F3709', 'L3_S30_F3714', 'L3_S30_F3719', 'L3_S30_F3724', 'L3_S30_F3729', 'L3_S30_F3734', 'L3_S30_F3739', 'L3_S30_F3744', 'L3_S30_F3749', 'L3_S30_F3754', 'L3_S30_F3759', 'L3_S30_F3764', 'L3_S30_F3769', 'L3_S30_F3774', 'L3_S30_F3794', 'L3_S30_F3804', 'L3_S30_F3809', 'L3_S30_F3829', 'L3_S33_F3855', 'L3_S33_F3859', 'L3_S33_F3865', 'L3_S33_F3867', 'L3_S33_F3869', 'L3_S33_F3871', 'L3_S33_F3873', 'L3_S34_F3876', 'L3_S34_F3878', 'L3_S34_F3880', 'L3_S34_F3882', 'L3_S35_F3884', 'L3_S35_F3889', 'L3_S35_F3894', 'L3_S35_F3898', 'L3_S35_F3903', 'L3_S35_F3908', 'L3_S35_F3913', 'L3_S36_F3918', 'L3_S36_F3920', 'L3_S36_F3926', 'L3_S36_F3930', 'L3_S36_F3934', 'L3_S36_F3938', 'L3_S37_F3944', 'L3_S37_F3946', 'L3_S37_F3948', 'L3_S37_F3950']

class SklearnClassifier:
    def __init__(self, model_path: Path | str) -> None:
        self.model = joblib.load(model_path)
        self.normalizer = StandardScaler()
        self.feature_names = np.array(FEAT_NAMES)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return self.normalizer.fit_transform(X)
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = pd.DataFrame(data=X.reshape((1, -1)), columns=self.feature_names)
        return self._predict(X)[0]