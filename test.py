import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tqdm.notebook import tqdm
import logging
import os

# Redirect print statements to a text file
output_file = f"Iterations/Iteration_{it}/Neural Network 6 (Trained on below Cohort) Training Logs.txt"
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(output_file, mode="w")  # Logs to file
    ]
)
logger = logging.getLogger()

num_features = len(binned_features)

param_grid = {
    'num_layers':None,
    'layer_sizes':[x for x in get_all_permutations([64,48,32,16,8])[5:] if len(x)<=3],
    'dropout_rates':None,
    'batch_size':[32, 64, 128],
    'epochs':[50],
    'learning_rate':[0.001]
}

def gini_score(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return 2*auc-1

param_combinations = []
for layer_sizes in param_grid['layer_sizes']:
    layers = len(layer_sizes)
    for batch_size in param_grid['batch_size']:
        if len(layer_sizes)>=layers:
            param_combinations.append(
                {
                    "num_layers": layers,
                    "layer_sizes": layer_sizes,
                    "dropout_rate":[0.2 for _ in range(layers)],
                    "batch_size":batch_size,
                }
            )

# Placeholder for results and best model
results = []
best_model_object1 = None
best_model_object2 = None
best_model_object3 = None
best_model1 = None  # Stores the best model's metrics and parameters
best_model2 = None
best_model3 = None
res_df_save_path = f"Iterations/Iteration_{it}/NN 6 Model (Trained on below Cohort) Training Progress.csv"

trial_number = 0
# Training loop
for params in tqdm(param_combinations, desc="Tuning Models"):
    trial_number += 1
    logger.info("\n-------------------------------------------------------------------------------------------------------------------------------------------")
    logger.info(f'\n ------------------ Trial {trial_number} --------------------------')
    # Build the model
    model = Sequential()
    model.add(Dense(params['layer_sizes'][0], input_dim=num_features, activation="relu"))
    model.add(Dropout(params['dropout_rate'][0]))
    for i in range(1, params['num_layers']):
        model.add(Dense(params['layer_sizes'][i], activation='relu'))
        model.add(Dropout(params['dropout_rate'][i]))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_auc', patience=4, mode='max', restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_auc', save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-6, verbose=1)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=params['batch_size'],
        epochs=param_grid['epochs'][0],
        callbacks=[early_stopping, checkpoint, reduce_lr],
        verbose=0
    )

    # Evaluate the model
    y_train_pred = model.predict(X_train).ravel()
    y_test_pred = model.predict(X_test).ravel()
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_gini = gini_score(y_train, y_train_pred)
    test_gini = gini_score(y_test, y_test_pred)
    auc_diff = abs(train_auc - test_auc)

    results.append({
        'params': params,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_gini': train_gini,
        'test_gini': test_gini,
        'auc_diff': auc_diff
    })

    res_df = pd.DataFrame(results)
    res_df.to_csv(res_df_save_path, mode='a', header = not os.path.exists(res_df_save_path), index=False)

    # Check if the current model is the best model
    if not best_model1 or ((test_auc > best_model1['test_auc'])):
        best_model1 = results[-1]
        best_model_object1 = model  # Save the current model object as the best
        with open(f"Iterations/Iteration_{it}/NN Model 6 Best Model 1 (Trained on below Cohort) on L4 Selected Features.pkl","wb") as file:
            pickle.dump(best_model_object1, file)

    if not best_model2 or ((test_auc > best_model2['test_auc']) & (auc_diff < 0.075)):
        best_model2 = results[-1]
        best_model_object2 = model  # Save the current model object as the best
        with open(f"Iterations/Iteration_{it}/NN Model 6 Best Model 2 (Trained on below Cohort) on L4 Selected Features.pkl","wb") as file:
            pickle.dump(best_model_object2, file)

    if not best_model3 or ((test_auc > best_model3['test_auc']) & (auc_diff < 0.09)):
        best_model3 = results[-1]
        best_model_object3 = model  # Save the current model object as the best
        with open(f"Iterations/Iteration_{it}/NN Model 6 Best Model 3 (Trained on below Cohort) on L4 Selected Features.pkl","wb") as file:
            pickle.dump(best_model_object3, file)

    # Display Progress
    logger.info(f"Params: {params}")
    logger.info(f"Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
    logger.info(f"Train Gini: {train_gini:.4f}, Test Gini: {test_gini:.4f}\n")

    # Sort results by Test AUC (descending) and AUC difference (ascending)
    sorted_results = sorted(results, key=lambda x: (-x['test_auc'], x['auc_diff']))

    logger.info('\n----------- Best Model 1 ------------')
    logger.info("Best Model Parameters: %s", best_model1['params'])
    logger.info(f"Best Train AUC: {best_model1['train_auc']:.4f}, Best Train Gini: {best_model1['train_gini']:.4f}")
    logger.info(f"Best Test AUC: {best_model1['test_auc']:.4f}, Best Test Gini: {best_model1['test_gini']:.4f}")
    logger.info(f"Train-Test AUC Difference: {best_model1['auc_diff']:.4f}")

    logger.info('\n----------- Best Model 2 ------------')
    logger.info("Best Model Parameters: %s", best_model2['params'])
    logger.info(f"Best Train AUC: {best_model2['train_auc']:.4f}, Best Train Gini: {best_model2['train_gini']:.4f}")
    logger.info(f"Best Test AUC: {best_model2['test_auc']:.4f}, Best Test Gini: {best_model2['test_gini']:.4f}")
    logger.info(f"Train-Test AUC Difference: {best_model2['auc_diff']:.4f}")

    logger.info('\n----------- Best Model 3 ------------')
    logger.info("Best Model Parameters: %s", best_model3['params'])
    logger.info(f"Best Train AUC: {best_model3['train_auc']:.4f}, Best Train Gini: {best_model3['train_gini']:.4f}")
    logger.info(f"Best Test AUC: {best_model3['test_auc']:.4f}, Best Test Gini: {best_model3['test_gini']:.4f}")
    logger.info(f"Train-Test AUC Difference: {best_model3['auc_diff']:.4f}")

# Final Output
logger.info(f"Model tuning output has been saved to {output_file}.")
