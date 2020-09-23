import streamlit as st
from plot_func import general_plot, local_plot
from utils import get_dataframe, get_shap_values, get_target, get_model_name, model_fit, get_row_number

st.set_option('deprecation.showfileUploaderEncoding', False)

st.markdown("<h1 style='text-align: center; '> Avec la Lab, attrapez les tous !</h1>",
            unsafe_allow_html=True)
data = get_dataframe()

target = get_target(data)

model_name = get_model_name()

model = model_fit(model_name, data, target)


st.info(" Shap is working hard! Please wait")

shap_values = get_shap_values(model_name, model, data)

st.info(" Thank you")

fig_general = general_plot(shap_values)
st.plotly_chart(fig_general, sharing='streamlit')

row = get_row_number()

fig_local = local_plot(int(row), shap_values)
st.plotly_chart(fig_local, sharing='streamlit')
