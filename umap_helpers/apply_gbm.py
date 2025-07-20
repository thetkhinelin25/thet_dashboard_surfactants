import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
import numpy as np

def syn_prediction(original_data, synthetic_primary_data, synthetic_secondary_data, synthetic_tertiary_data):
    synthetic_data = pd.concat([synthetic_primary_data, synthetic_secondary_data, synthetic_tertiary_data], ignore_index=True)

    feature_cols = [
        'primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name',
        'primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)'
    ]

    regression_targets = [
        'initial ph', 'appearance absorption value', 'height foam (mm)', 'mildness'
    ]

    classification_targets = ['clarity', 'colour', 'physical state', 'cmc']

    categorical_cols = ['primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name']
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    label_encoders = {}
    combined_features = pd.concat([original_data[feature_cols], synthetic_data[feature_cols]], axis=0)

    for col in categorical_cols:
        le = LabelEncoder()
        combined_features[col] = le.fit_transform(combined_features[col].astype(str))
        label_encoders[col] = le

    original_data[feature_cols] = combined_features.iloc[:len(original_data)].values
    synthetic_data[feature_cols] = combined_features.iloc[len(original_data):].values

    for col in classification_targets:
        le = LabelEncoder()
        original_data[col] = le.fit_transform(original_data[col].astype(str))
        label_encoders[col] = le

    scaler_X = StandardScaler()
    scaler_Y_dict = {target: StandardScaler() for target in regression_targets}

    original_data[numeric_cols] = scaler_X.fit_transform(original_data[numeric_cols])
    synthetic_data[numeric_cols] = scaler_X.transform(synthetic_data[numeric_cols])
    for target in regression_targets:
        original_data[[target]] = scaler_Y_dict[target].fit_transform(original_data[[target]])

    X_train = original_data[feature_cols]
    X_synth = synthetic_data[feature_cols]
    Y_train_clf = original_data[classification_targets]

    # -- Regression model --
    Y_pred_reg_dict = {}

    # 1. RF for 'height foam (mm)'
    reg = RandomForestRegressor(random_state=99)
    grid = GridSearchCV(reg, {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }, cv=KFold(n_splits=3, shuffle=True, random_state=9), n_jobs=-1, scoring='r2')
    grid.fit(X_train, original_data[['height foam (mm)']])
    Y_pred_reg_dict['height foam (mm)'] = grid.predict(X_synth)

    # 2. Use GBM for remaining: initial ph', 'mildness', 'appearance absorption value'
    gbm_targets = ['initial ph', 'mildness', 'appearance absorption value']
    reg = GradientBoostingRegressor(random_state=99)
    grid = GridSearchCV(reg, {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }, cv=KFold(n_splits=3, shuffle=True, random_state=9), n_jobs=-1, scoring='r2')
    multi_reg_model = MultiOutputRegressor(grid)
    multi_reg_model.fit(X_train, original_data[gbm_targets])
    gbm_preds = multi_reg_model.predict(X_synth)
    for i, target in enumerate(gbm_targets):
        Y_pred_reg_dict[target] = gbm_preds[:, i]


    # -- Classification model --
    # Use GBM for 'clarity', 'colour', 'physical state', 'cmc'
    classifier = GradientBoostingClassifier(random_state=99)
    clf_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [7, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'subsample': [1.0]
    }
    clf_grid = GridSearchCV(classifier, clf_param_grid, cv=KFold(n_splits=3, shuffle=True, random_state=9), n_jobs=-1, scoring='balanced_accuracy')
    multi_clf_model = MultiOutputClassifier(clf_grid)
    multi_clf_model.fit(X_train, Y_train_clf)
    Y_pred_clf = multi_clf_model.predict(X_synth)

    Y_pred_clf_df = pd.DataFrame(Y_pred_clf, columns=classification_targets)
    for col in classification_targets:
        Y_pred_clf_df[col] = label_encoders[col].inverse_transform(Y_pred_clf_df[col])

    Y_pred_reg_df = pd.DataFrame()
    for col in regression_targets:
        pred = np.array(Y_pred_reg_dict[col]).reshape(-1, 1)
        Y_pred_reg_df[col] = scaler_Y_dict[col].inverse_transform(pred).ravel()

    synthetic_output = synthetic_data.copy()
    synthetic_output[numeric_cols] = scaler_X.inverse_transform(synthetic_output[numeric_cols])

    for col in categorical_cols:
        synthetic_output[col] = label_encoders[col].inverse_transform(synthetic_output[col].astype(int))

    synthetic_output[regression_targets] = Y_pred_reg_df
    synthetic_output[classification_targets] = Y_pred_clf_df

    return synthetic_output
