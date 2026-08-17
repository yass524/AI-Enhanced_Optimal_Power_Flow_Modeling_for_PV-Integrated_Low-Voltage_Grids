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
from joblib import load
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
file_path1 = "/Users/yassinsaad/Desktop/bachelor thesis/data/train_3w_final.csv"
file_path3 = "/Users/yassinsaad/Desktop/bachelor thesis/data/Paper2_tr_ts/test_3w_paper2style_A.csv"
file_path4 = "/Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_paper2style-2.csv"
file_path5 = "/Users/yassinsaad/Desktop/bachelor thesis/data/test_3w_newntwk-4.csv"
old_data_tr = pd.read_csv(file_path1)
new_data1_ts = pd.read_csv(file_path3)
new_data2_ts = pd.read_csv(file_path4)
new_data3_ts = pd.read_csv(file_path5)
X_train, y_train, feature_names, target_names, index_train, customers_order = prepare_wide(old_data_tr)
X_test_new1, y_test_new1, _, _, index_test, _ = prepare_wide(new_data1_ts, customers_order=customers_order)
X_test_new2, y_test_new2, _, _, index_test_new2, _ = prepare_wide(new_data2_ts, customers_order=customers_order)
X_test_neww, y_test_neww, _, _, index_test_neww, _ = prepare_wide(new_data3_ts, customers_order=customers_order)

xscale = MinMaxScaler().fit(X_train)
yscale = MinMaxScaler().fit(y_train)
x_test_scaled_new1= xscale.transform(X_test_new1)
y_test_scaled_new1 = yscale.transform(y_test_new1)

#%%
split = int(0.9 * len(x_test_scaled_new1))
X_tr, X_val = x_test_scaled_new1[:split], x_test_scaled_new1[split:]
y_tr, y_val = y_test_scaled_new1[:split], y_test_scaled_new1[split:]

def eval_metrics(model, X_scaled, y_true_scaled, yscale):
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    # back to Volts for reporting
    y_pred = yscale.inverse_transform(y_pred_scaled)
    y_true = yscale.inverse_transform(y_true_scaled)
    rmse_global = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    avg_abs     = float(np.mean(np.abs(y_pred - y_true)))
    max_abs     = float(np.max(np.abs(y_pred - y_true)))
    test_MAE = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    test_R2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    n_samples = y_true.shape[0]
    n_features = X_scaled.shape[1]
    if n_samples > n_features + 1 and not np.isnan(test_R2):
        r2_adj = float(1 - (1 - test_R2) * (n_samples - 1) / (n_samples - n_features - 1))
    else:
        r2_adj = float("nan")

    return rmse_global, avg_abs, max_abs, test_MAE, test_R2, r2_adj

def print_report(tag, model, X_tr, y_tr, X_va, y_va, yscale, t_start,):
    tr = eval_metrics(model, X_tr, y_tr, yscale)
    va = eval_metrics(model, X_va, y_va, yscale)
    dt = time.perf_counter() - t_start
    print(f"\n[{tag}]  time: {dt:.1f}s")
    print(
        f"  Train -> RMSE: {tr[0]:.3f} V | avg |err|: {tr[1]:.3f} V | "
        f"max |err|: {tr[2]:.3f} V | MAE: {tr[3]:.3f} V | R2: {tr[4]:.4f} | R2adj: {tr[5]:.4f}"
    )
    print(
        f"  Valid -> RMSE: {va[0]:.3f} V | avg |err|: {va[1]:.3f} V | "
        f"max |err|: {va[2]:.3f} V | MAE: {va[3]:.3f} V | R2: {va[4]:.4f} | R2adj: {va[5]:.4f}"
    )

early = callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")
#%%
# Transfer learning on NN 1
best_model_1layer = load_model("nn_star.h5", compile=False)
for i, L in enumerate(best_model_1layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))

for l in best_model_1layer.layers[:-1]:
    l.trainable = False
best_model_1layer.layers[-1].trainable = True

best_model_1layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
t0 = time.perf_counter()
best_model_1layer.fit(X_tr, y_tr, validation_data=(X_val, y_val),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-only_NN1L",best_model_1layer, X_tr, y_tr, X_val, y_val, yscale, t0)


best_model_1layer.save("nn_star_head_tuned.h5")
# %%
#plots
y_val_pred_s = best_model_1layer.predict(X_val, verbose=0)
y_val_pred   = yscale.inverse_transform(y_val_pred_s)
y_val_true   = yscale.inverse_transform(y_val)

plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_val_true[:500,0], label="True V")
plt.plot(index_test[:500], y_val_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 1)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# %%
#transfer learning on NN 1 new data 2
best_model_1layer = load_model("nn_star.h5", compile=False)
for l in best_model_1layer.layers[:-1]:
    l.trainable = False
best_model_1layer.layers[-1].trainable = True
best_model_1layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new2_scaled= xscale.transform(X_test_new2)
y_test_new2_scaled = yscale.transform(y_test_new2)
split2 = int(0.9 * len(X_test_new2_scaled))
X_tr2, X_val2 = X_test_new2_scaled[:split2], X_test_new2_scaled[split2:]
y_tr2, y_val2 = y_test_new2_scaled[:split2], y_test_new2_scaled[split2:]
t02 = time.perf_counter()
best_model_1layer.fit(X_tr2, y_tr2, validation_data=(X_val2, y_val2),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-only_NN1L_newdata2", best_model_1layer, X_tr2, y_tr2, X_val2, y_val2, yscale, t02)
best_model_1layer.save("nn_star_head_tuned_newdata2.h5")

#plots
y_val2_pred_s = best_model_1layer.predict(X_val2, verbose=0)
y_val2_pred   = yscale.inverse_transform(y_val2_pred_s)
y_val2_true   = yscale.inverse_transform(y_val2)
plt.figure(figsize=(10,4))
plt.plot(index_test_new2[:500], y_val2_true[:500,0], label="True V")
plt.plot(index_test_new2[:500], y_val2_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 1, new data 2)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
#transfer learning on NN 1 new data 3 (largest variance) ultimate test
best_model_1layer = load_model("nn_star.h5", compile=False)
for l in best_model_1layer.layers[:-1]:
    l.trainable = False
best_model_1layer.layers[-1].trainable = True
best_model_1layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new3_scaled= xscale.transform(X_test_neww)
y_test_new3_scaled = yscale.transform(y_test_neww)
split3 = int(0.9 * len(X_test_new3_scaled))
X_tr3, X_val3 = X_test_new3_scaled[:split3], X_test_new3_scaled[split3:]
y_tr3, y_val3 = y_test_new3_scaled[:split3], y_test_new3_scaled[split3:]
t03 = time.perf_counter()
best_model_1layer.fit(X_tr3, y_tr3, validation_data=(X_val3, y_val3),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-only_NN1L_newdata3", best_model_1layer, X_tr3, y_tr3, X_val3, y_val3, yscale, t03)
best_model_1layer.save("nn_star_head_tuned_newdata3.h5")

#plots
y_val3_pred_s = best_model_1layer.predict(X_val3, verbose=0)
y_val3_pred   = yscale.inverse_transform(y_val3_pred_s)
y_val3_true   = yscale.inverse_transform(y_val3)
plt.figure(figsize=(10,4))
plt.plot(index_test_neww[:500], y_val3_true[:500,0], label="True V")
plt.plot(index_test_neww[:500], y_val3_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 1, new data 3)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# Transfer learning on NN 2 layer with new data 1
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True

best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
t1 = time.perf_counter()
best_model_2layer.fit(X_tr, y_tr, validation_data=(X_val, y_val),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])

print_report("Head-and-last-H_NN2L", best_model_2layer, X_tr, y_tr, X_val, y_val, yscale, t1)
best_model_2layer.save("nn_2layer_star_head_tuned.h5")

#plots
y_val_pred2_s = best_model_2layer.predict(X_val, verbose=0)
y_val_pred2   = yscale.inverse_transform(y_val_pred2_s)
y_val_true2   = yscale.inverse_transform(y_val)
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_val_true2[:500,0], label="True V")
plt.plot(index_test[:500], y_val_pred2[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
# %%
# Transfer learning on NN 2 layer (head only)
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
#best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True

best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
t1 = time.perf_counter()
best_model_2layer.fit(X_tr, y_tr, validation_data=(X_val, y_val),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])

print_report("Head-ONLY_NN2L", best_model_2layer, X_tr, y_tr, X_val, y_val, yscale, t1)
best_model_2layer.save("nn_2layer_star_head_only.h5")

#plots
y_val_pred2_s = best_model_2layer.predict(X_val, verbose=0)
y_val_pred2   = yscale.inverse_transform(y_val_pred2_s)
y_val_true2   = yscale.inverse_transform(y_val)
plt.figure(figsize=(10,4))
plt.plot(index_test[:500], y_val_true2[:500,0], label="True V")
plt.plot(index_test[:500], y_val_pred2[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# Transfer learning on NN 2 layer with new data 2
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True
best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new2_scaled= xscale.transform(X_test_new2)
y_test_new2_scaled = yscale.transform(y_test_new2)
split2 = int(0.9 * len(X_test_new2_scaled))
X_tr2, X_val2 = X_test_new2_scaled[:split2], X_test_new2_scaled[split2:]
y_tr2, y_val2 = y_test_new2_scaled[:split2], y_test_new2_scaled[split2:]
t02 = time.perf_counter()
best_model_2layer.fit(X_tr2, y_tr2, validation_data=(X_val2, y_val2),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-and-last-H_NN2L_newdata2", best_model_2layer, X_tr2, y_tr2, X_val2, y_val2, yscale, t02)
best_model_2layer.save("nn_2layer_star_head_tuned_newdata2.h5")
#plots
y_val2_pred_s = best_model_2layer.predict(X_val2, verbose=0)
y_val2_pred   = yscale.inverse_transform(y_val2_pred_s)
y_val2_true   = yscale.inverse_transform(y_val2)
plt.figure(figsize=(10,4))
plt.plot(index_test_new2[:500], y_val2_true[:500,0], label="True V")
plt.plot(index_test_new2[:500], y_val2_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2, new data 2)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# transfer learning on NN 2 layer withe new data 2 head only
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
#best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True
best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new2_scaled= xscale.transform(X_test_new2)
y_test_new2_scaled = yscale.transform(y_test_new2)
split2 = int(0.9 * len(X_test_new2_scaled))
X_tr2, X_val2 = X_test_new2_scaled[:split2], X_test_new2_scaled[split2:]
y_tr2, y_val2 = y_test_new2_scaled[:split2], y_test_new2_scaled[split2:]
t02 = time.perf_counter()
best_model_2layer.fit(X_tr2, y_tr2, validation_data=(X_val2, y_val2),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-ONLY_NN2L_newdata2", best_model_2layer, X_tr2, y_tr2, X_val2, y_val2, yscale, t02)
best_model_2layer.save("nn_2layer_star_head_only_newdata2.h5")
#plots
y_val2_pred_s = best_model_2layer.predict(X_val2, verbose=0 )
y_val2_pred   = yscale.inverse_transform(y_val2_pred_s)
y_val2_true   = yscale.inverse_transform(y_val2)
plt.figure(figsize=(10,4))
plt.plot(index_test_new2[:500], y_val2_true[:500,0], label="True V")
plt.plot(index_test_new2[:500], y_val2_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2, new data 2)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# transfer learning on NN 2 layer with new data 3 head only
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
#best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True
best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new3_scaled= xscale.transform(X_test_neww)
y_test_new3_scaled = yscale.transform(y_test_neww)
split3 = int(0.9 * len(X_test_new3_scaled))
X_tr3, X_val3 = X_test_new3_scaled[:split3], X_test_new3_scaled[split3:]
y_tr3, y_val3 = y_test_new3_scaled[:split3], y_test_new3_scaled[split3:]
t03 = time.perf_counter()
best_model_2layer.fit(X_tr3, y_tr3, validation_data=(X_val3, y_val3),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-ONLY_NN2L_newdata3", best_model_2layer, X_tr3, y_tr3, X_val3, y_val3, yscale, t03)
best_model_2layer.save("nn_2layer_star_head_only_newdata3.h5")
#plots
y_val3_pred_s = best_model_2layer.predict(X_val3, verbose=0)
y_val3_pred   = yscale.inverse_transform(y_val3_pred_s)
y_val3_true   = yscale.inverse_transform(y_val3)
plt.figure(figsize=(10,4))
plt.plot(index_test_neww[:500], y_val3_true[:500,0], label="True V")
plt.plot(index_test_neww[:500], y_val3_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2, new data 3)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
#%%
# Transfer learning on NN 2 layer with new data 3
best_model_2layer = load_model("nn_2layer_star.h5", compile=False)
for i, L in enumerate(best_model_2layer.layers):
    print(i, L.name, L.__class__.__name__, getattr(L, "units", None))
for l in best_model_2layer.layers[:-1]:
    l.trainable = False
best_model_2layer.layers[-2].trainable = True
best_model_2layer.layers[-1].trainable = True
best_model_2layer.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
X_test_new3_scaled= xscale.transform(X_test_neww)
y_test_new3_scaled = yscale.transform(y_test_neww)
split3 = int(0.9 * len(X_test_new3_scaled))
X_tr3, X_val3 = X_test_new3_scaled[:split3], X_test_new3_scaled[split3:]
y_tr3, y_val3 = y_test_new3_scaled[:split3], y_test_new3_scaled[split3:]
t03 = time.perf_counter()
best_model_2layer.fit(X_tr3, y_tr3, validation_data=(X_val3, y_val3),
         epochs=300, batch_size=48, shuffle=False, verbose=0, callbacks=[early])
print_report("Head-and-last-H_NN2L_newdata3", best_model_2layer, X_tr3, y_tr3, X_val3, y_val3, yscale, t03)
best_model_2layer.save("nn_2layer_star_head_tuned_newdata3.h5")
#plots
y_val3_pred_s = best_model_2layer.predict(X_val3, verbose=0)
y_val3_pred   = yscale.inverse_transform(y_val3_pred_s)
y_val3_true   = yscale.inverse_transform(y_val3)
plt.figure(figsize=(10,4))
plt.plot(index_test_neww[:500], y_val3_true[:500,0], label="True V")
plt.plot(index_test_neww[:500], y_val3_pred[:500,0], label="Predicted V", alpha=0.7)
plt.title("Customer 1 Voltage – Test Data (transfer learning on NN 2, new data 3)");
plt.xlabel("Time"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()










# %%
print(X_test_new1.shape, X_test_new2.shape, X_test_neww.shape)


# %%
model = load_model("nn_star_head_tuned.h5", compile=False)
W1, b1 = model.layers[1].get_weights()   # tanh layer, shape (295,1168) & (1168,)
W2, b2 = model.layers[2].get_weights()   # output layer, shape (1168,146) & (146,)

np.savetxt("W1_tl.csv", W1, delimiter=",")
np.savetxt("b1_tl.csv", b1, delimiter=",")
np.savetxt("W2_tl.csv", W2, delimiter=",")
np.savetxt("b2_tl.csv", b2, delimiter=",")
# after prepare_wide(...) on Paper-2 train set
np.savetxt("customers_order_paper2.csv", np.array(customers_order, dtype=str), fmt="%s", delimiter=",")# %%

# %%




# make sure this is the latest prepare_wide (the one you sent me)
from transfer_learning import prepare_wide

# 1) Load ONLY the Paper-2 train CSV
train_p2 = pd.read_csv("/Users/yassinsaad/Desktop/bachelor thesis/data/train_3w_paper2style_A.csv")

# 2) Let prepare_wide choose the order (customers_order=None)
X_p2, y_p2, fn_p2, tn_p2, idx_p2, customers_order_p2 = prepare_wide(
    train_p2,
    customers_order=None,   # IMPORTANT: let it compute the order
    ts_col="timestamp",
    id_col="customer_id",
    p_col="P_kW",
    q_col="Q_kVAR",
    vtx_col="V_tx_V",
    vnode_col="V_node_V",
)

print("n_x should be 2*H + 3 =", X_p2.shape[1])
print("H (customers) =", len(customers_order_p2))

# 3) Sanity check: must be 146 customers and n_x = 2*146+3 = 295
assert len(customers_order_p2) == 146, "customers_order_p2 must have 146 entries"
assert X_p2.shape[1] == 295, "input dimension must be 295 (2*146+3)"

# 4) Save as a clean CSV (overwrite old file)
np.savetxt(
    "customers_order_paper2.csv",
    np.array(customers_order_p2, dtype=str),
    fmt="%s",
    delimiter=",",
)
print("Saved customers_order_paper2.csv with", len(customers_order_p2), "IDs")
# %%
m = load_model("nn_star_head_tuned.h5", compile=False)
W1, b1 = m.layers[1].get_weights()
print("W1 shape:", W1.shape)   # should be (295, 1168)
# %%
np.savetxt("customers_order_paper2.csv", np.array(customers_order, dtype=int), fmt="%d", delimiter=",")
# %%
