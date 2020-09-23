import pandas as pd
import plotly.graph_objects as go
import numpy as np


def general_plot(shap_values):
    df_means = pd.DataFrame()

    y_axis_means = list(shap_values.feature_names)
    means = list(np.abs(shap_values.values).mean(axis=0))
    means = [round(mean, 3) for mean in means]

    df_means['y'] = y_axis_means
    df_means['mean'] = means
    df_means = df_means.sort_values(by='y', ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=df_means['mean'],
        y=df_means['y'],
        name="Impacts sur le score",
        orientation='h',
        marker_color="Navy",
        text=df_means['mean'],
        opacity=0.7),

    ],
    )
    fig.update_traces(
        textposition='auto',
        textfont_size=12)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        title={
            'text': "Impact Absolu Global",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    return fig


def local_plot(row, shap_values):

    x_axis = list(shap_values[0].values)
    base_x = shap_values[row].base_values

    y_axis = ["{} = {}".format(name, val)for (name, val) in zip(
        shap_values.feature_names, list(shap_values[row].data))]
    base_y = "moyenne"
    df = pd.DataFrame()

    x_axis = [round(x, 3) for x in x_axis]
    df['x'] = x_axis
    df['y'] = y_axis

    base_x = shap_values[0].base_values
    base_y = "Moyenne du modèle"
    df2 = pd.DataFrame()
    df2['x'] = [round(base_x, 3)]
    df2['y'] = [base_y]

    df = df.sort_values(by='y', ascending=False)
    df = df2.append(df)
    df["Color"] = np.where(df["x"] < 0, 'darkred', 'RoyalBlue')
    fig = go.Figure(data=[go.Bar(
        x=df['x'],
        y=df['y'],
        name="Impacts sur le score",
        orientation='h',
        marker_color=df['Color'],
        text=df['x'],
        opacity=0.7),

    ],
    )
    fig.update_traces(
        textposition='auto',
        textfont_size=12)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        title={
            'text': "Impact Local",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    return fig
