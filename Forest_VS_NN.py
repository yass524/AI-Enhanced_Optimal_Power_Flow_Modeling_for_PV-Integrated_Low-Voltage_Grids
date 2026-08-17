#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

import time
#%%


# Function to prepare data in wide format (make features in array form for each timestamp like in paper)


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
# Load data
file_path1 = "/Users/yassinsaad/Desktop/bachelor thesis/data/train_3w_final.csv"
file_path2 = "//Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_final.csv"
data_train = pd.read_csv(file_path1)
data_test = pd.read_csv(file_path2)
X_train, y_train, feature_names, target_names, index_train, customers_order = prepare_wide(data_train)
X_test, y_test, _, _, index_test, _ = prepare_wide(data_test,customers_order=customers_order)

print(X_train.shape)
print(y_train.shape)
X_train, y_train, feature_names, target_names, index_train, customers_order = prepare_wide(data_train)
#%%
# --- PREVIEW processed data ---
# Combine first few rows of X and y into one DataFrame for inspection
preview_df = pd.DataFrame(
    np.hstack([X_train[:5], y_train[:5]]),
    columns = feature_names + target_names,
    index = index_train[:5]
)
print("\n=== SAMPLE OF PROCESSED TRAIN DATA (first 5 timestamps) ===")
print(preview_df.round(3))   # rounded for readability
print(f"\nShape of X: {X_train.shape},  y: {y_train.shape}")

print("\n=== ALL FEATURE NAMES (first 20 + last 10) ===")
print(feature_names[:20], "...", feature_names[-10:])  # show start & end

print("\n=== LAST COLUMNS OF PREVIEW (includes Vtr) ===")
print(preview_df.iloc[:, -10:].round(3))  # last 10 columns, includes Vtr and some Vnode

print("\n=== SAMPLE OF Q FEATURES (just one timestamp) ===")
print(preview_df.filter(like='Q_', axis=1).iloc[0, :10])  # first 10 Q columns for 1st timestamp

# %%
## Random Forest
RF = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)
rf_time = time.time()
RF.fit(X_train, y_train)
print(f"RF training time: {time.time() - rf_time:.2f} seconds")
y_pred = RF.predict(X_test)
rmse_rf_train = np.sqrt(((RF.predict(X_train) - y_train)**2).mean(axis=0))
avg_rmse_rf_train = rmse_rf_train.mean()
print(f"RF  avg RMSE from train: {avg_rmse_rf_train:.3f} V")

rmse_rf_per_cust = np.sqrt(((y_pred - y_test)**2).mean(axis=0))
avg_rmse_rf = rmse_rf_per_cust.mean()
max_abs_rf = np.max(np.abs(y_pred - y_test))
print(f"RF  avg RMSE from test: {avg_rmse_rf:.3f} V | max |err|: {max_abs_rf:.3f} V")

#plotting predicted vs actual for a sample customer
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test[:500,0], label="True V")
plt.plot(index_test[:500], y_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (RF)")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
# %%
#------------------------------------------------------------------------------------------------------------------------
#
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras import layers
from keras import models, optimizers, regularizers
from keras import callbacks, backend as K

# --- Scaling ---
xscale = MinMaxScaler()
yscale = MinMaxScaler()
X_train_scaled = xscale.fit_transform(X_train)
X_test_scaled  = xscale.transform(X_test)
y_train_scaled = yscale.fit_transform(y_train)
y_test_scaled  = yscale.transform(y_test)
#%%
# --- Inspect scaled data (compact view across feature types + outputs) ---
preview_rows = min(5, X_train_scaled.shape[0])
preview_customers = customers_order[:min(3, len(customers_order))]  # show first few customers

x_scaled_df = pd.DataFrame(
    X_train_scaled[:preview_rows],
    columns=feature_names,
    index=index_train[:preview_rows]
)
y_scaled_df = pd.DataFrame(
    y_train_scaled[:preview_rows],
    columns=target_names,
    index=index_train[:preview_rows]
)

combined_preview = pd.DataFrame(index=x_scaled_df.index)
for cid in preview_customers:
    p_col = f'P_{cid}'
    q_col = f'Q_{cid}'
    v_col = f'Vnode_{cid}'
    if p_col in x_scaled_df.columns:
        combined_preview[p_col] = x_scaled_df[p_col]
    if q_col in x_scaled_df.columns:
        combined_preview[q_col] = x_scaled_df[q_col]
    if v_col in y_scaled_df.columns:
        combined_preview[v_col] = y_scaled_df[v_col]

for vt_col in ['Vtr_A', 'Vtr_B', 'Vtr_C']:
    if vt_col in x_scaled_df.columns:
        combined_preview[vt_col] = x_scaled_df[vt_col]

print("\n=== SCALED preview (first 5 rows; P/Q/Vnode for first few customers + Vtr) ===")
print(combined_preview.round(3))
#%%
n_y = y_train.shape[1]  # 146
n_x = X_train.shape[1]  # 295
hidden_units = 8 * n_y  # 8 × 146 = 1168
#%%
# --- Model definition ---
np.random.seed(42)
tf.random.set_seed(42)

def build_model():
    X = keras.Input(shape=(n_x,))
    H = layers.Dense(hidden_units, activation='tanh')(X)
    Y = layers.Dense(n_y, activation='linear')(H)
    model = keras.Model(inputs=X, outputs=Y)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

# --- Train 10 models like in the paper ---
rmse_list = []
models_list = []

best_model = None
best_tr_rmse = np.inf

for i in range(10):
    np.random.seed(100 + i)
    tf.random.set_seed(100 + i)
    K.clear_session()

    model = build_model()
    model.fit(X_train_scaled, y_train_scaled,
              epochs=1000, batch_size=48,
              shuffle=False, verbose=0)

    y_pred_train_s = model.predict(X_train_scaled, verbose=0)
    y_pred_train   = yscale.inverse_transform(y_pred_train_s)
    tr_rmse = np.sqrt(((y_pred_train - y_train)**2).mean())  # in volts

    rmse_list.append(tr_rmse)
    models_list.append(model)
    print(f"Run {i+1}: Train RMSE = {tr_rmse:.4f} V")

    if tr_rmse < best_tr_rmse:
        best_tr_rmse = tr_rmse
        best_model = model

# --- Display summary ---
print("\n=== Summary of 10 runs ===")
for i, val in enumerate(rmse_list, start=1):
    marker = " <== BEST" if val == best_tr_rmse else ""
    print(f"Run {i:2d}: {val:.4f} V{marker}")
print(f"\nBest model RMSE (V): {best_tr_rmse:.4f}")

# --- Evaluate best model on test set ---
y_pred_nn_scaled = best_model.predict(X_test_scaled, verbose=0)
y_pred_nn = yscale.inverse_transform(y_pred_nn_scaled)

y_tr_best_s = best_model.predict(X_train_scaled, verbose=0)
y_tr_best   = yscale.inverse_transform(y_tr_best_s)

rmse_nn_train = np.sqrt(((y_tr_best - y_train)**2).mean())
avg_rmse_nn_train = rmse_nn_train.mean()
print(f"\nNN  RMSE from train: {rmse_nn_train:.3f} V")

rmse_nn_per_cust = np.sqrt(((y_pred_nn - y_test)**2).mean())
avg_rmse_nn = rmse_nn_per_cust.mean()
avg_abs_nn = np.mean(np.abs(y_pred_nn - y_test))
max_abs_nn = np.max(np.abs(y_pred_nn - y_test))
print(f"NN  RMSE: {rmse_nn_per_cust:.3f} V | max |err|: {max_abs_nn:.3f} V | avg |err|:  {avg_abs_nn:.3f} V")
# plotting predicted vs actual for a sample customer
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_nn[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (NN)")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# --- Save the best model ---
best_model.save("nn_star.h5")
print("\nBest model saved as nn_star.h5")
# Save MinMaxScaler parameters (we'll reproduce the same transform in MATLAB)
np.savetxt("x_min.csv", xscale.data_min_, delimiter=",")
np.savetxt("x_max.csv", xscale.data_max_, delimiter=",")
np.savetxt("y_min.csv", yscale.data_min_, delimiter=",")
np.savetxt("y_max.csv", yscale.data_max_, delimiter=",")

# Save customers_order used in training (very important)
np.savetxt("customers_order.csv", np.array(customers_order, dtype=int), fmt="%d", delimiter=",")

print("Saved: nn_star.h5, x_min/max.csv, y_min/max.csv, customers_order.csv")
# %%
# NN with regularization
#hidden_units2 = 2 * n_y # 4 * 146 = 584
X_new = keras.Input(shape=(n_x,))
H_new = layers.Dense(hidden_units, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(X_new)
Y_new = layers.Dense(n_y, activation='linear')(H_new)
K.clear_session()
model2 = keras.Model(inputs=X_new, outputs=Y_new)
model2.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')

model2.fit(X_train_scaled, y_train_scaled, epochs=1000, batch_size=48, shuffle=False, verbose=0)

y_tr2_s = model2.predict(X_train_scaled, verbose=0)
y_tr2   = yscale.inverse_transform(y_tr2_s)

y_pred_nn_scaled_reg = model2.predict(X_test_scaled,verbose=0)
y_pred_nn_reg = yscale.inverse_transform(y_pred_nn_scaled_reg)

rmse_nn_train_reg = np.sqrt(((y_tr2 - y_train)**2).mean(axis=0))
avg_rmse_nn_train_reg = rmse_nn_train_reg.mean()
print(f"NN with regularization avg RMSE from train: {avg_rmse_nn_train_reg:.3f} V")

rmse_nn_per_cust_reg = np.sqrt(((y_pred_nn_reg - y_test)**2).mean(axis=0))
avg_rmse_nn_reg = rmse_nn_per_cust_reg.mean()
max_abs_nn_reg = np.max(np.abs(y_pred_nn_reg - y_test))
print(f"NN with regularization avg RMSE: {avg_rmse_nn_reg:.3f} V | max |err|: {max_abs_nn_reg:.3f} V")

#plot predicted vs actual for a sample customer
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_nn_reg[:500,0], label="Pred   icted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (NN with regularization)")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# %%
# ---------- Metrics per-customer ----------
err = y_pred_nn_reg - y_test                           # (T, H)
rmse_per_cust = np.sqrt((err**2).mean(axis=0))         # (H,)
worst_idx = int(np.argmax(rmse_per_cust))
median_idx = int(np.argsort(rmse_per_cust)[len(rmse_per_cust)//2])

def plot_customer(idx, title_suffix=""):
    cid = customers_order[idx]
    plt.figure(figsize=(11,4))
    plt.plot(index_test, y_test[:, idx], label="True Voltage", linewidth=1.2)
    plt.plot(index_test, y_pred_nn_reg[:, idx], label="Predicted Voltage", linewidth=1.0, alpha=0.85)
    plt.title(f"Customer {cid} – Predicted vs Actual {title_suffix}")
    plt.xlabel("Time")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save an image for slides
    outpath = f"pred_vs_actual_{cid}.png"
    plt.savefig(outpath, dpi=220)
    print(f"Saved: {outpath}")

# Plot worst and (optionally) a median customer
plot_customer(worst_idx, title_suffix=f"(worst RMSE = {rmse_per_cust[worst_idx]:.2f} V)")
# plot_customer(median_idx, title_suffix=f\"(median RMSE = {rmse_per_cust[median_idx]:.2f} V)\")

# ---------- Top-10 biggest absolute errors (any timestamp, any customer) ----------
abs_err = np.abs(err)                                  # (T, H)
T, H = abs_err.shape
k = 10
flat_idx = np.argpartition(abs_err.ravel(), -(k))[-k:] # indices of k largest errors (unsorted)
# Sort descending
flat_idx = flat_idx[np.argsort(abs_err.ravel()[flat_idx])[::-1]]

rows, cols = np.unravel_index(flat_idx, (T, H))
top10_df = pd.DataFrame({
    "rank": np.arange(1, k+1),
    "timestamp": index_test.values[rows],
    "customer_id": [customers_order[c] for c in cols],
    "actual_V": y_test[rows, cols],
    "predicted_V": y_pred_nn_reg[rows, cols],
    "abs_error_V": abs_err[rows, cols]
})
print(top10_df)

# Optional: save to CSV for your appendix/slide
top10_df.to_csv("top10_largest_errors_nn.csv", index=False)
print("Saved: top10_largest_errors_nn.csv")



# %%
# second paper's data
from keras.models import load_model
file_path3 = "/Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_paper2style-2.csv"
data_test2 = pd.read_csv(file_path3)
X_test2, y_test2, _, _, index_test2, _ = prepare_wide(data_test2,customers_order=customers_order)

X_test2_scaled  = xscale.transform(X_test2)
#%%#testing with second paper's data using best model from first paper
bestt = load_model("nn_star.h5", compile=False)
y_pred_nn2_scaled = bestt.predict(X_test2_scaled, verbose=0)
y_pred_nn2 = yscale.inverse_transform(y_pred_nn2_scaled)
rmse_nn2 = np.sqrt(((y_pred_nn2 - y_test2)**2).mean())
avg_dev2 = np.mean(np.abs(y_pred_nn2 - y_test2))
max_dev2 = np.max(np.abs(y_pred_nn2 - y_test2))

print(f" NN on Paper-2-style Test Data")
print(f"RMSE: {rmse_nn2:.3f} V | Avg |err|: {avg_dev2:.3f} V | Max |err|: {max_dev2:.3f} V")

plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test2[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_nn2[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Paper-2-style Test Data")
plt.xlabel("Time")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%
#second paper's data test 2
file_path4 ="/Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_paper2style_A.csv"
data_test3 = pd.read_csv(file_path4)
X_test3, y_test3, _, _, index_test3, _ = prepare_wide(data_test3,customers_order=customers_order)
X_test3_scaled  = xscale.transform(X_test3)

#%%
y_pred_nn3_scaled = bestt.predict(X_test3_scaled, verbose=0)
y_pred_nn3 = yscale.inverse_transform(y_pred_nn3_scaled)
rmse_nn3 = np.sqrt(((y_pred_nn3 - y_test3)**2).mean())
avg_dev3 = np.mean(np.abs(y_pred_nn3 - y_test3))
max_dev3 = np.max(np.abs(y_pred_nn3 - y_test3))
print(f" NN on Paper-2-style Test Data")
print(f"RMSE: {rmse_nn3:.3f} V | Avg |err|: {avg_dev3:.3f} V | Max |err|: {max_dev3:.3f} V")
plt.figure(figsize=(10,4))
plt.plot(index_test3[:500], y_test3[:500,0], label="True V")
plt.plot(index_test3[:500], y_pred_nn3[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Paper-2-style Test-2 Data")
plt.xlabel("Time")
plt.ylabel("Voltage [V]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%


#hidden_units = 8 * n_y  # 8 × 146 = 1168 neurons
hidden_units2 = 6 * n_y  # 6 * 146 = 876 neurons
hidden_units3 = 4 * n_y  # 4 * 146 = 584 neurons
hidden_units4 = 2 * n_y  # 2 * 146 = 292 neurons

X_deep = keras.Input(shape=(n_x,))
H1_deep = layers.Dense(hidden_units,  activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(X_deep)
H2_deep = layers.Dense(hidden_units2, activation='tanh', kernel_regularizer=regularizers.l2(1e-4))(H1_deep)
H3_deep = layers.Dense(hidden_units3, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(H2_deep)
H4_deep = layers.Dense(hidden_units4, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(H3_deep)
Y_deep  = layers.Dense(n_y, activation='linear')(H4_deep)

model_deep = keras.Model(X_deep, Y_deep)
model_deep.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')

model_deep.fit(X_train_scaled, y_train_scaled,
               epochs=3000, batch_size=48, shuffle=True, verbose=0)

# --- Evaluate on training data ---
y_tr_deep_s = model_deep.predict(X_train_scaled, verbose=0)
y_tr_deep   = yscale.inverse_transform(y_tr_deep_s)

# --- Evaluate on same LV network ---
y_pred_deep_s = model_deep.predict(X_test_scaled, verbose=0)
y_pred_deep   = yscale.inverse_transform(y_pred_deep_s)

rmse_nn_test_deep = np.sqrt(((y_pred_deep - y_test)**2).mean())
max_abs_nn_deep   = np.max(np.abs(y_pred_deep - y_test))
avg_dev_nn_deep   = np.mean(np.abs(y_pred_deep - y_test))

print(f"NN Deep RMSE (same feeder): {rmse_nn_test_deep:.3f} V | Avg dev: {avg_dev_nn_deep:.3f} V | Max |err|: {max_abs_nn_deep:.3f} V")

# Plot same feeder
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_test[:500,0], label="True V")
plt.plot(index_test[:500], y_pred_deep[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (same LV)")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# --- Paper-2-style Test 1 ---
y_pred_deep2_s = model_deep.predict(X_test2_scaled, verbose=0)
y_pred_deep2   = yscale.inverse_transform(y_pred_deep2_s)

rmse_nn2_deep = np.sqrt(((y_pred_deep2 - y_test2)**2).mean())
avg_dev2_deep = np.mean(np.abs(y_pred_deep2 - y_test2))
max_dev2_deep = np.max(np.abs(y_pred_deep2 - y_test2))

print(f"\nNN Deep on Paper-2-style Test 1:")
print(f"RMSE: {rmse_nn2_deep:.3f} V | Avg |err|: {avg_dev2_deep:.3f} V | Max |err|: {max_dev2_deep:.3f} V")

plt.figure(figsize=(10,4))
plt.plot(index_test2[:500], y_test2[:500,0], label="True V")
plt.plot(index_test2[:500], y_pred_deep2[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Paper-2-style Test 1")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# --- Paper-2-style Test 2 ---
y_pred_deep3_s = model_deep.predict(X_test3_scaled, verbose=0)
y_pred_deep3   = yscale.inverse_transform(y_pred_deep3_s)

rmse_nn3_deep = np.sqrt(((y_pred_deep3 - y_test3)**2).mean())
avg_dev3_deep = np.mean(np.abs(y_pred_deep3 - y_test3))
max_dev3_deep = np.max(np.abs(y_pred_deep3 - y_test3))

print(f"\nNN Deep on Paper-2-style Test 2:")
print(f"RMSE: {rmse_nn3_deep:.3f} V | Avg |err|: {avg_dev3_deep:.3f} V | Max |err|: {max_dev3_deep:.3f} V")

plt.figure(figsize=(10,4))
plt.plot(index_test3[:500], y_test3[:500,0], label="True V")
plt.plot(index_test3[:500], y_pred_deep3[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Paper-2-style Test 2")
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# %%
# The 2 layer NN
hidden_units3 = 4 * n_y # 4 * 146 = 584 neurons
def build_2layer_model():
    X = keras.Input(shape=(n_x,))
    H1 = layers.Dense(hidden_units, activation='tanh')(X)
    H2 = layers.Dense(hidden_units3, activation='tanh')(H1)
    Y = layers.Dense(n_y, activation='linear')(H2)
    model = keras.Model(inputs=X, outputs=Y)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model
# --- Train 10 models like in the paper ---
rmse_list_2layer = []
models_list_2layer = []
best_model_2layer = None
best_tr_rmse_2layer = np.inf
for i in range(10):
    np.random.seed(200 + i)
    tf.random.set_seed(200 + i)
    K.clear_session()

    model_2layer = build_2layer_model()
    model_2layer.fit(X_train_scaled, y_train_scaled,
              epochs=1000, batch_size=48,
              shuffle=False, verbose=0)

    y_pred_train_s_2layer = model_2layer.predict(X_train_scaled, verbose=0)
    y_pred_train_2layer   = yscale.inverse_transform(y_pred_train_s_2layer)
    tr_rmse_2layer = np.sqrt(((y_pred_train_2layer - y_train)**2).mean())  # in volts

    rmse_list_2layer.append(tr_rmse_2layer)
    models_list_2layer.append(model_2layer)
    print(f"2-Layer Run {i+1}: Train RMSE = {tr_rmse_2layer:.4f} V")

    if tr_rmse_2layer < best_tr_rmse_2layer:
        best_tr_rmse_2layer = tr_rmse_2layer
        best_model_2layer = model_2layer

# --- Display summary ---
print("\n=== Summary of 10 runs for 2-layer NN ===")
for i, val in enumerate(rmse_list_2layer, start=1):
    marker = " <== BEST" if val == best_tr_rmse_2layer else ""
    print(f"Run {i:2d}: {val:.4f} V{marker}")
print(f"\nBest 2-layer model RMSE (V): {best_tr_rmse_2layer:.4f}")

# --- Evaluate best 2-layer model on test set ---
y_pred_nn_scaled_2layer = best_model_2layer.predict(X_test_scaled, verbose=0)
y_pred_nn_2layer = yscale.inverse_transform(y_pred_nn_scaled_2layer)

y_tr_best_s_2layer = best_model_2layer.predict(X_train_scaled, verbose=0)
y_tr_best_2layer   = yscale.inverse_transform(y_tr_best_s_2layer)

rmse_nn_train_2layer = np.sqrt(((y_tr_best_2layer - y_train)**2).mean())
avg_rmse_nn_train_2layer = rmse_nn_train_2layer.mean()
print(f"\n2-Layer NN  RMSE from train: {rmse_nn_train_2layer:.3f} V")

rmse_nn_per_cust_2layer = np.sqrt(((y_pred_nn_2layer - y_test)**2).mean())
avg_rmse_nn_2layer = rmse_nn_per_cust_2layer.mean()
avg_abs_nn_2layer = np.mean(np.abs(y_pred_nn_2layer - y_test))
max_abs_nn_2layer = np.max(np.abs(y_pred_nn_2layer - y_test))
print(f"2-Layer NN  RMSE: {rmse_nn_per_cust_2layer:.3f} V | max |err|: {max_abs_nn_2layer:.3f} V | avg |err|:  {avg_abs_nn_2layer:.3f} V")

# --- Save the best 2-layer model ---
best_model_2layer.save("nn_2layer_star.h5")
print("\nBest 2-layer model saved as nn_2layer_star.h5")
# %%
from keras.models import load_model

# Load WITHOUT compiling (ignore loss, metrics, optimizer)
try:
    # Keras 3 supports safe_mode; TF-Keras 2.x does not
    model = load_model("nn_star.h5", compile=False, safe_mode=False)
except TypeError:
    model = load_model("nn_star.h5", compile=False)

# (Optional) sanity check layers (robust to layers without output_shape)
for i, layer in enumerate(model.layers):
    out_shape = getattr(layer, "output_shape", None) or getattr(layer, "batch_output_shape", None)
    print(i, layer.name, layer.__class__.__name__, out_shape)

# Extract only Dense layers to avoid index mismatch if the model changes
dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
if len(dense_layers) < 2:
    raise ValueError(f"Expected at least 2 Dense layers (hidden + output); found {len(dense_layers)}")

W1, b1 = dense_layers[0].get_weights()   # Dense(tanh)
W2, b2 = dense_layers[1].get_weights()   # Dense(linear)

# Save to CSV for MATLAB
np.savetxt("W1.csv", W1, delimiter=",")
np.savetxt("b1.csv", b1, delimiter=",")
np.savetxt("W2.csv", W2, delimiter=",")
np.savetxt("b2.csv", b2, delimiter=",")

print("Saved W1.csv, b1.csv, W2.csv, b2.csv")

# %%
