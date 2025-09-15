"""
Temporal Fusion Transformer (TFT) Attention and Interpretability Visualizations

This module provides functions to extract and visualize attention weights and variable 
selection information from a trained PyTorch Forecasting TFT model. It uses the model's raw 
predictions to create interactive Plotly charts for:
    1. Encoder attention over past timesteps (per-sample and average)
    2. Decoder (future) attention
    3. Variable selection weights (static, encoder, decoder)
    4. Attention evolution across prediction horizons
    5. Multi-head attention patterns

Usage: Obtain raw outputs via 
    raw_output, x = model.predict(dataloader, mode="raw", return_x=True)
then pass `raw_output` (and optionally the `model` for variable names) to the functions below.
"""
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_encoder_attention(raw_output, horizon=0, sample_index=0):
    """
    Plots encoder attention over past time steps for a given prediction horizon.
    - raw_output: dict from mode='raw' predict (contains 'encoder_attention').
    - horizon: future step index to visualize (default 0 for first prediction).
    - sample_index: index of sample in batch.
    Returns: Plotly Figure of attention vs. encoder time steps, with one line per head plus an average line.
    """
    # something like: raw_output["encoder_attention"]: (batch, heads, dec_len, enc_len)
    encoder_att = raw_output["encoder_attention"].cpu().detach().numpy()
    batch_size, n_heads, dec_len, enc_len = encoder_att.shape
    if horizon >= dec_len:
        raise ValueError(f"horizon {horizon} out of range (max {dec_len-1})")

    att_sample = encoder_att[sample_index, :, horizon, :]
    att_avg    = np.nanmean(att_sample, axis=0)

    fig = go.Figure()

    # per-head attention traces idk
    for h in range(n_heads):
        fig.add_trace(go.Scatter(
            x=np.arange(enc_len), y=att_sample[h],
            mode='lines', name=f"Head {h}",
            line=dict(width=2),
            hovertemplate=f"Head {h}<br>Time step: %{{x}}<br>Weight: %{{y:.4f}}"
        ))
    # average attention trace, maybe?
    fig.add_trace(go.Scatter(
        x=np.arange(enc_len), y=att_avg,
        mode='lines', name="Average",
        line=dict(color='black', width=4, dash='dash'),
        hovertemplate="Average<br>Time step: %{x}<br>Weight: %{y:.4f}"
    ))
    fig.update_layout(
        title=f"Encoder Attention (Sample {sample_index}, Horizon {horizon})",
        xaxis_title="Past Time Steps (Encoder)",
        yaxis_title="Attention Weight",
        legend_title="Attention Heads",
        hovermode="x unified"
    )
    return fig

def plot_decoder_attention(raw_output, horizon=0, sample_index=0):
    """
    Plots decoder (self-)attention over future steps for a given prediction horizon.
    - raw_output: dict from mode='raw' predict (contains 'decoder_attention').
    - horizon: future step index (default 0).
    - sample_index: index of sample in batch.
    Returns: Plotly Figure of decoder attention weights.
    """
    # raw_output["decoder_attention"]: (batch, heads, dec_len, dec_len)
    decoder_att = raw_output["decoder_attention"].cpu().detach().numpy()
    batch_size, n_heads, dec_len, _ = decoder_att.shape
    if horizon >= dec_len:
        raise ValueError(f"horizon {horizon} out of range (max {dec_len-1})")
    # extract for chosen sample and horizon: shape = (heads, dec_len)
    att_sample = decoder_att[sample_index, :, horizon, :]
    att_avg    = np.nanmean(att_sample, axis=0)
    fig = go.Figure()
    for h in range(n_heads):
        fig.add_trace(go.Scatter(
            x=np.arange(dec_len), y=att_sample[h],
            mode='lines', name=f"Head {h}",
            line=dict(width=2),
            hovertemplate=f"Head {h}<br>Future step: %{{x}}<br>Weight: %{{y:.4f}}"
        ))
    fig.add_trace(go.Scatter(
        x=np.arange(dec_len), y=att_avg,
        mode='lines', name="Average",
        line=dict(color='black', width=4, dash='dash'),
        hovertemplate="Average<br>Future step: %{x}<br>Weight: %{y:.4f}"
    ))
    fig.update_layout(
        title=f"Decoder Attention (Sample {sample_index}, Horizon {horizon})",
        xaxis_title="Future Time Steps (Decoder)",
        yaxis_title="Attention Weight",
        legend_title="Attention Heads",
        hovermode="x unified"
    )
    return fig

def plot_variable_selection(raw_output, model=None):
    """
    Plots variable selection weights for static, encoder, and decoder variables.
    If `model` (TFT) is given, uses model.hparams names; otherwise uses generic labels.
    Returns a Plotly Figure with three subplots (static / encoder / decoder) and a sample-index slider.
    """
    # extract weights and lengths from those etc
    static_w  = raw_output["static_variables"].cpu().detach().numpy().squeeze()  # (batch, n_static)
    encoder_w = raw_output["encoder_variables"].cpu().detach().numpy().squeeze() # (batch, enc_len, n_encoder_vars)
    decoder_w = raw_output["decoder_variables"].cpu().detach().numpy().squeeze() # (batch, dec_len, n_decoder_vars)
    enc_len   = raw_output["encoder_lengths"].cpu().detach().numpy()  # (batch,)
    dec_len   = raw_output["decoder_lengths"].cpu().detach().numpy()  # (batch,)
    batch_size = static_w.shape[0]

    # determine variable names from model.hparams if available i guess
    if model is not None:
        static_names = list(getattr(model.hparams, "static_categoricals", [])) + list(getattr(model.hparams, "static_reals", []))
        enc_names    = list(getattr(model.hparams, "time_varying_categoricals_encoder", [])) + list(getattr(model.hparams, "time_varying_reals_encoder", []))
        dec_names    = list(getattr(model.hparams, "time_varying_categoricals_decoder", [])) + list(getattr(model.hparams, "time_varying_reals_decoder", []))
    else:
        static_names = [f"static_{i}" for i in range(static_w.shape[1])]
        enc_names    = [f"enc_{i}" for i in range(encoder_w.shape[2])]
        dec_names    = [f"dec_{i}" for i in range(decoder_w.shape[2])]

    # aggregate encoder/decoder weights across time (mean over valid time steps)
    encoder_mean = np.zeros((batch_size, len(enc_names)))
    decoder_mean = np.zeros((batch_size, len(dec_names)))
    for i in range(batch_size):
        if encoder_w.ndim == 3:
            if enc_len[i] > 0:
                encoder_mean[i] = np.nanmean(encoder_w[i, :enc_len[i], :], axis=0)
        else:
            encoder_mean[i] = encoder_w[i]
        if decoder_w.ndim == 3:
            if dec_len[i] > 0:
                decoder_mean[i] = np.nanmean(decoder_w[i, :dec_len[i], :], axis=0)
        else:
            decoder_mean[i] = decoder_w[i]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Static Variables", "Encoder Variables", "Decoder Variables"))

    # static variable bars
    for j, name in enumerate(static_names):
        fig.add_trace(go.Bar(x=[name], y=[static_w[0, j]], name=name), row=1, col=1)
    # encoder variable bars
    for j, name in enumerate(enc_names):
        fig.add_trace(go.Bar(x=[name], y=[encoder_mean[0, j]], name=name), row=1, col=2)
    # decoder variable bars
    for j, name in enumerate(dec_names):
        fig.add_trace(go.Bar(x=[name], y=[decoder_mean[0, j]], name=name), row=1, col=3)

    # slider to select sample index
    steps = []
    for i in range(batch_size):
        y_static = [float(static_w[i, j]) for j in range(len(static_names))]
        y_enc    = [float(encoder_mean[i, j]) for j in range(len(enc_names))]
        y_dec    = [float(decoder_mean[i, j]) for j in range(len(dec_names))]
        step = {"method": "update", "label": str(i), "args": [{"y": []}]}
        step["args"][0]["y"] = [[v] for v in (y_static + y_enc + y_dec)]
        steps.append(step)
    sliders = [dict(active=0, currentvalue={"prefix": "Sample: "}, pad={"t": 30}, steps=steps)]
    fig.update_layout(title="Variable Selection Weights (per sample)", sliders=sliders, showlegend=False)
    fig.update_yaxes(title_text="Importance", row=1, col=1)
    fig.update_yaxes(title_text="Importance", row=1, col=2)
    fig.update_yaxes(title_text="Importance", row=1, col=3)
    return fig

def plot_attention_evolution(raw_output, sample_index=0, head=None):
    """
    Plots encoder attention evolution across all prediction horizons as a heatmap.
    - raw_output: dict from mode='raw' predict.
    - sample_index: which sample in batch to visualize.
    - head: optional head index (default None = average over heads).
    Returns: Plotly Figure (heatmap) with x=past time, y=horizon.
    """
    # raw_output["encoder_attention"]: (batch, heads, dec_len, enc_len)
    encoder_att = raw_output["encoder_attention"].cpu().detach().numpy()
    _, n_heads, dec_len, enc_len = encoder_att.shape
    att_sample = encoder_att[sample_index]  # shape = (heads, dec_len, enc_len)
    if head is None:
        att_matrix = np.nanmean(att_sample, axis=0)  # (dec_len, enc_len)
        title_suffix = "Avg of Heads"
    else:
        att_matrix = att_sample[head]               # (dec_len, enc_len)
        title_suffix = f"Head {head}"
    fig = go.Figure(data=go.Heatmap(
        z=att_matrix, x=np.arange(enc_len), y=np.arange(dec_len),
        colorscale="Viridis", colorbar_title="Attention"
    ))
    fig.update_layout(
        title=f"Encoder Attention Evolution (Sample {sample_index}, {title_suffix})",
        xaxis_title="Past Time Steps", yaxis_title="Prediction Horizon"
    )
    return fig
