"""
modelli/base.py
================
Componenti condivisi da tutte le strategie di training: il wrapper che
rende un booster XGBoost compatibile con l'interfaccia scikit-learn
(predict + feature_names_in_), e la FACTORY della loss custom con
penalità di consistenza.

Perché una factory e non una funzione fissa: la loss custom deve "vedere"
i dati di un dataset specifico (X_vicino, peso_arco_arr), che cambiano da
un training all'altro (standard, anelli, 6 anelli, regionale...). Nel
notebook originale questo veniva risolto duplicando la funzione con un
suffisso diverso per ogni training (loss_consistenza_xgb_anelli,
loss_consistenza_xgb_6anelli, ...) — qui usiamo una closure, che crea una
nuova funzione "su misura" senza duplicare codice.
"""

import time

import numpy as np
import xgboost as xgb


class WrapperXGBoost:
    """
    Rende un Booster XGBoost compatibile con l'interfaccia scikit-learn,
    cosi' può essere usato come `model` in predizioni.genera_predizioni
    senza modifiche a quella funzione.
    """

    def __init__(self, booster: xgb.Booster, feature_names: list[str]):
        self.booster = booster
        self.feature_names_in_ = np.array(feature_names)

    def predict(self, X):
        X_ordinato = X[self.feature_names_in_] if hasattr(X, "columns") else X
        return self.booster.predict(xgb.DMatrix(X_ordinato))


def crea_loss_consistenza(
    peso_arco_arr: np.ndarray, x_vicino_train, lambda_consistenza: float = 0.5
):
    """
    Factory che costruisce una loss custom per XGBoost, parametrizzata sui
    dati del training corrente. Restituisce (loss_fn, callback_class) pronti
    da passare a xgb.train(obj=loss_fn, callbacks=[callback_class()]).

    Formulazione (per ogni esempio i, un nodo u_i rispetto a un target):

        L_i = 0.5*(h(u_i) - tempo_reale_i)^2
            + lambda * max(0, h(u_i) - w(u_i,v_i) - h(v_i))^2

    dove v_i e' un vicino noto di u_i (un arco reale del grafo), w(u_i,v_i)
    e' il peso reale di quell'arco, e h(v_i) e' la predizione CORRENTE del
    modello per il vicino (aggiornata ad ogni round di boosting tramite il
    callback restituito).

    Gradiente e hessiano analitici (verificati numericamente con differenze
    finite prima dell'uso):
        dL/dh(u_i)  = (h(u_i) - tempo_reale_i) + 2*lambda*violazione_i
        d2L/dh(u_i)^2 = 1 + 2*lambda   [se violazione_i > 0, altrimenti 1]

    Parametri:
        peso_arco_arr      : array (n_esempi,) con il peso reale dell'arco
                             u_i -> v_i per ogni riga del train set
        x_vicino_train     : DataFrame (n_esempi, n_feature) con le feature
                             del vicino v_i per ogni riga (stesso ordine di
                             peso_arco_arr)
        lambda_consistenza : peso della penalità di consistenza nella loss

    Restituisce:
        (loss_fn, AggiornaBoosterCallback)
    """

    state = {"booster_attuale": None}

    def loss_fn(y_pred, dtrain):
        y_true = dtrain.get_label()

        booster_corrente = state["booster_attuale"]
        if booster_corrente is not None:
            d_vicino = xgb.DMatrix(x_vicino_train.fillna(0))
            h_vicino = booster_corrente.predict(d_vicino)
        else:
            h_vicino = np.zeros_like(y_pred)  # primo round: nessun riferimento ancora

        soglia = peso_arco_arr + h_vicino
        violazione = np.maximum(0, y_pred - soglia)
        violazione[np.isinf(soglia)] = 0  # righe senza vicino: penalità disattivata

        grad = (y_pred - y_true) + 2 * lambda_consistenza * violazione
        hess = np.ones_like(y_pred) + 2 * lambda_consistenza * (violazione > 0).astype(float)

        return grad, hess

    class AggiornaBoosterCallback(xgb.callback.TrainingCallback):
        """
        Aggiorna il riferimento al booster corrente dopo ogni round di
        boosting, cosi' la loss custom puo' calcolare h(v) con le
        predizioni piu' recenti. Stampa anche un avanzamento periodico,
        perché con la loss custom (che richiama predict() ad ogni round)
        il training può richiedere diversi minuti.
        """

        def __init__(self, ogni_n_round: int = 20, n_round_totali: int | None = None):
            self.ogni_n_round = ogni_n_round
            self.n_round_totali = n_round_totali
            self.t0 = time.time()

        def after_iteration(self, model, epoch, evals_log):
            state["booster_attuale"] = model
            if epoch == 0 or (epoch + 1) % self.ogni_n_round == 0:
                trascorso = time.time() - self.t0
                totale_str = f"/{self.n_round_totali}" if self.n_round_totali else ""
                print(f"  Round {epoch + 1}{totale_str}  ({trascorso:.1f}s trascorsi)")
            return False  # False = continua il training

    return loss_fn, AggiornaBoosterCallback
