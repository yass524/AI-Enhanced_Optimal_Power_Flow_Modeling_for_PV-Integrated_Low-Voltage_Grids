#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers, regularizers
from keras import callbacks, backend as K
from keras.models import load_model
import time
#%%
def prepare_wide(
    df: pd.DataFrame,
    customers_order=None,
    drop_incomplete=True,
    tx_phase_col='phase',          # set to None if you don't trust the phase labels
    ts_col='timestamp',
    id_col='customer_id',
    p_col='P_kW',
    q_col='Q_kVAR',
    vtx_col='V_tx_V',
    vnode_col='V_node_V'
):
    """
    Convert long-format feeder data into paper-style wide matrices.
    Returns: X (np.ndarray), y (np.ndarray), feature_names (list), target_names (list), index (DatetimeIndex/Index)
    """

    # Ensure timestamp is sorted and usable as index later
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        try:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception:
            pass
    df = df.sort_values(ts_col)

    # Fix customer order (stable across splits)
    if customers_order is None:
        customers_order = (
            df[[id_col, tx_phase_col] if tx_phase_col in df.columns else [id_col]]
            .drop_duplicates(subset=[id_col])
            .sort_values(id_col)[id_col]
            .tolist()
        )

    # Pivot P, Q, and V_node to wide (timestamps x customers)
    P_wide = df.pivot(index=ts_col, columns=id_col, values=p_col).reindex(columns=customers_order)
    Q_wide = df.pivot(index=ts_col, columns=id_col, values=q_col).reindex(columns=customers_order)
    Vnode_wide = df.pivot(index=ts_col, columns=id_col, values=vnode_col).reindex(columns=customers_order)

    # Transformer voltages: try to get 3 phases, else replicate single value
    if tx_phase_col in df.columns and df[tx_phase_col].notna().any():
        Vtr_phase = (
            df.pivot_table(index=ts_col, columns=tx_phase_col, values=vtx_col, aggfunc='first')
              .rename(columns={'A':'Vtr_A','B':'Vtr_B','C':'Vtr_C'})
        )
        # if some phases missing, fill by forward/backward fill or fallback to row mean
        if not {'Vtr_A','Vtr_B','Vtr_C'}.issubset(set(Vtr_phase.columns)):
            for miss in {'Vtr_A','Vtr_B','Vtr_C'} - set(Vtr_phase.columns):
                Vtr_phase[miss] = np.nan
        Vtr_phase = Vtr_phase[['Vtr_A','Vtr_B','Vtr_C']]
        # gentle fill strategy
        Vtr_phase = Vtr_phase.ffill().bfill()
        # if still NaNs, fill with row mean
        row_mean = Vtr_phase.mean(axis=1)
        Vtr_phase = Vtr_phase.apply(lambda s: s.fillna(row_mean))
    else:
        # one V_tx per timestamp → replicate to 3 columns
        Vtr_single = df.groupby(ts_col, as_index=True)[vtx_col].first()
        Vtr_phase = pd.DataFrame(
            {'Vtr_A': Vtr_single, 'Vtr_B': Vtr_single, 'Vtr_C': Vtr_single},
            index=Vtr_single.index
        )

    # Align indices (timestamps common to all)
    common_idx = P_wide.index
    for mat in (Q_wide, Vnode_wide, Vtr_phase):
        common_idx = common_idx.intersection(mat.index)

    P_wide = P_wide.loc[common_idx]
    Q_wide = Q_wide.loc[common_idx]
    Vnode_wide = Vnode_wide.loc[common_idx]
    Vtr_phase = Vtr_phase.loc[common_idx]

    if drop_incomplete:
        # Drop any timestamps with NaNs (e.g., missing customer rows)
        mask = (~P_wide.isna().any(axis=1) &
                ~Q_wide.isna().any(axis=1) &
                ~Vnode_wide.isna().any(axis=1) &
                ~Vtr_phase.isna().any(axis=1))
        P_wide, Q_wide, Vnode_wide, Vtr_phase = P_wide[mask], Q_wide[mask], Vnode_wide[mask], Vtr_phase[mask]
        common_idx = P_wide.index

    # Build feature matrix X: [P_all | Q_all | Vtr_A | Vtr_B | Vtr_C]
    feature_names = (
        [f'P_{cid}' for cid in customers_order] +
        [f'Q_{cid}' for cid in customers_order] +
        ['Vtr_A','Vtr_B','Vtr_C']
    )
    X = np.hstack([P_wide.values, Q_wide.values, Vtr_phase[['Vtr_A','Vtr_B','Vtr_C']].values])

    target_names = [f'Vnode_{cid}' for cid in customers_order]
    y = Vnode_wide.values

    return X, y, feature_names, target_names, common_idx, customers_order
#%%
file_path1 = "/Users/yassinsaad/Desktop/bachelor thesis/data/train_3w_final.csv"
file_path2 = "/Users/yassinsaad/Desktop/bachelor thesis/data/Paper2_tr_ts/train_3w_paper2style_A.csv"
file_path3 = "/Users/yassinsaad/Desktop/bachelor thesis/data/Paper2_tr_ts/test_3w_paper2style_A.csv"
file_path4 = "/Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_paper2style-2.csv"
old_data_tr = pd.read_csv(file_path1)
new_data1_tr = pd.read_csv(file_path2)
new_data1_ts = pd.read_csv(file_path3)
new_data2_ts = pd.read_csv(file_path4)

X_train, y_train, feature_names, target_names, index_train, customers_order = prepare_wide(old_data_tr)
X_train_new1, y_train_new1, _, _, index_train_new1, _ = prepare_wide(new_data1_tr, customers_order=customers_order)
X_test_new1, y_test_new1, _, _, index_test, _ = prepare_wide(new_data1_ts, customers_order=customers_order)
X_test_new2, y_test_new2, _, _, index_test_new2, _ = prepare_wide(new_data2_ts, customers_order=customers_order)

#%%
#train and test new data_1 on NN and deep NN (keep track of error and time)
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
X_train_scaled_new1 = xscaler.fit_transform(X_train_new1)
y_train_scaled_new1 = yscaler.fit_transform(y_train_new1)
x_test_scaled_new1= xscaler.transform(X_test_new1)
y_test_scaled_new1 = yscaler.transform(y_test_new1)

n_x_new1 = X_train_scaled_new1.shape[1]
n_y_new1 = y_train_scaled_new1.shape[1]
hidden_units = 8 * n_y_new1  # 8 × 146 = 1168

X = keras.Input(shape=(n_x_new1,))
H = layers.Dense(hidden_units, activation='tanh')(X)
Y = layers.Dense(n_y_new1, activation='linear')(H)
model = keras.Model(inputs=X, outputs=Y)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
start_time = time.time()
model.fit(X_train_scaled_new1, y_train_scaled_new1,epochs=1000, batch_size=48,shuffle=False, verbose=0)
train_time = time.time() - start_time
print(f"Training time (NN): {train_time:.2f} s")

y_pred_scaled_new1_tr = model.predict(X_train_scaled_new1,verbose=0)
y_pred_new1_tr = yscaler.inverse_transform(y_pred_scaled_new1_tr)
rmse_new1_tr= np.sqrt(np.mean((y_train_new1 - y_pred_new1_tr) ** 2))
print("Train RMSE on new data 1:", rmse_new1_tr)

y_pred_scaled_new1 = model.predict(x_test_scaled_new1,verbose=0)
y_pred_new1 = yscaler.inverse_transform(y_pred_scaled_new1)
rmse_new1= np.sqrt(np.mean((y_test_new1 - y_pred_new1) ** 2))
print("Test RMSE on new data 1:", rmse_new1)

plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test_new1[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_new1[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage - Paper-2-style Test Data (NN)")
plt.xlabel("Time")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#train on new data 1 and test on new data 2
X_test_new2_scaled= xscaler.transform(X_test_new2)
y_test_new2_scaled = yscaler.transform(y_test_new2)

y_pred_scaled_new2 = model.predict(X_test_new2_scaled,verbose=0)
y_pred_new2 = yscaler.inverse_transform(y_pred_scaled_new2)
rmse_new2= np.sqrt(np.mean((y_test_new2 - y_pred_new2) ** 2))
print("Test RMSE on new data 2:", rmse_new2)

plt.figure(figsize=(10,4))
plt.plot(index_test_new2[:500], y_test_new2[:500,0], label="True V")
plt.plot(index_test_new2[:500], y_pred_new2[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage - Paper-2B-style Test 2 Data (training on A) (NN)")
plt.xlabel("Time")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#DNN

hidden_units2 = 6 * n_y_new1  # 6 * 146 = 876 neurons
hidden_units3 = 4 * n_y_new1  # 4 * 146 = 584 neurons
hidden_units4 = 2 * n_y_new1 # 2 * 146 = 292 neurons

X_deep = keras.Input(shape=(n_x_new1,))
H1_deep = layers.Dense(hidden_units,  activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(X_deep)
H2_deep = layers.Dense(hidden_units2, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(H1_deep)
H3_deep = layers.Dense(hidden_units3, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(H2_deep)
H4_deep = layers.Dense(hidden_units4, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(H3_deep)
Y_deep  = layers.Dense(n_y_new1, activation='linear')(H4_deep)

model_deep = keras.Model(X_deep, Y_deep)
model_deep.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
start_time_deep = time.time()
model_deep.fit(X_train_scaled_new1, y_train_scaled_new1, epochs=3000, batch_size=48, shuffle=True, verbose=0)
train_time_deep = time.time() - start_time_deep
print(f"Training time (Deep NN): {train_time_deep:.2f} s")

y_tr_deep_s = model_deep.predict(X_train_scaled_new1, verbose=0)
y_tr_deep   = yscaler.inverse_transform(y_tr_deep_s)
rmse_train_deep = np.sqrt(((y_tr_deep - y_train_new1)**2).mean())
print("Train RMSE on new data 1 (Deep NN):", rmse_train_deep)

y_pred_deep_s = model_deep.predict(x_test_scaled_new1, verbose=0)
y_pred_deep   = yscaler.inverse_transform(y_pred_deep_s)
rmse_nn_test_deep = np.sqrt(((y_pred_deep - y_test_new1)**2).mean())
print("Test RMSE on new data 1 (Deep NN):", rmse_nn_test_deep)

plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test_new1[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_deep[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Paper-2-style Test-2 Data")
plt.xlabel("Time")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%
# --------------------------
# DATA PREPARATION
# --------------------------
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()

X_train_scaled = xscaler.fit_transform(X_train)
y_train_scaled = yscaler.fit_transform(y_train)
X_test_scaled  = xscaler.transform(X_test_new1)
y_test_scaled  = yscaler.transform(y_test_new1)

n_x = X_train_scaled.shape[1]
n_y = y_train_scaled.shape[1]

# --------------------------
# TRAIN MULTIPLE DEPTHS
# --------------------------
depth_results = []  # store (depth, train_time, train_RMSE, test_RMSE)

for depth in range(2, 9):  # from 2 to 8 hidden layers
    print(f"\n🧠 Training model with {depth} hidden layers...")

    # Define base layer width scaling (like in your Deep NN)
    base_units = 8 * n_y
    units = [int(base_units / (2 ** i)) for i in range(depth)]  # decreasing layer size

    # Build model dynamically
    inputs = layers.Input(shape=(n_x,))
    x = inputs
    for i, u in enumerate(units):
        activation = 'tanh' if i < 2 else 'relu'  # first layers tanh, then ReLU
        x = layers.Dense(u, activation=activation, kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(n_y, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')

    # Train and measure time
    start = time.time()
    model.fit(X_train_scaled, y_train_scaled, epochs=1500, batch_size=48, shuffle=True, verbose=0)
    train_time = time.time() - start

    # Evaluate
    y_train_pred_s = model.predict(X_train_scaled, verbose=0)
    y_train_pred = yscaler.inverse_transform(y_train_pred_s)
    train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))

    y_test_pred_s = model.predict(X_test_scaled, verbose=0)
    y_test_pred = yscaler.inverse_transform(y_test_pred_s)
    test_rmse = np.sqrt(np.mean((y_test_new1 - y_test_pred) ** 2))

    depth_results.append((depth, train_time, train_rmse, test_rmse))
    print(f"Layers: {depth} | Time: {train_time:.1f}s | Train RMSE: {train_rmse:.3f} V | Test RMSE: {test_rmse:.3f} V")

# --------------------------
# RESULTS SUMMARY
# --------------------------
depth_results = np.array(depth_results, dtype=object)
print("\n📊 Summary (Depth, Time [s], Train RMSE [V], Test RMSE [V]):")
for d, t, tr, te in depth_results:
    print(f"{d:>2} layers | {t:>7.1f} s | Train: {tr:.3f} V | Test: {te:.3f} V")

# --------------------------
# PLOTS
# --------------------------
depths = [r[0] for r in depth_results]
train_rmse_vals = [r[2] for r in depth_results]
test_rmse_vals = [r[3] for r in depth_results]
times = [r[1] for r in depth_results]

fig, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(depths, train_rmse_vals, 'o-', label='Train RMSE', color='green')
ax1.plot(depths, test_rmse_vals, 'o--', label='Test RMSE', color='red')
ax1.set_xlabel('Number of Hidden Layers')
ax1.set_ylabel('RMSE [V]')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(depths, times, 's--', label='Training Time [s]', color='blue')
ax2.set_ylabel('Training Time [s]')
ax2.legend(loc='upper right')

plt.title('Effect of Network Depth on Performance and Training Time')
plt.tight_layout()
plt.show()
# %%
