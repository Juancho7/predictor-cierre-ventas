# Página web para usar el modelo: ¿este cliente va a comprar?
# Para correrla:  streamlit run app.py
# Necesita en la misma carpeta: modelo_rf.joblib, columnas.joblib y referencia_test.joblib
# (los genera el notebook)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Cargamos el modelo y la lista de columnas que se usó al entrenar
modelo = joblib.load('modelo_rf.joblib')
columnas = joblib.load('columnas.joblib')
# Puntajes de los clientes del conjunto de prueba y si compraron o no (para comparar)
ref = joblib.load('referencia_test.joblib')
PUNTAJES_TEST = ref['puntajes']
COMPRO_TEST = ref['compro']
TASA_BASE = COMPRO_TEST.mean()   # % de clientes de prueba que compraron (~11.7%)

st.set_page_config(page_title='Predictor de cierre de ventas', page_icon=None)
st.title('¿Este cliente va a comprar?')
st.write('Llena los datos del cliente **antes de llamarlo** y el modelo (Random Forest) '
         'lo ubica en la lista de prioridad de llamadas.')

# Las opciones se muestran en español, pero el modelo las conoce en inglés (como vienen en el dataset)
TRABAJOS = {'Administrativo': 'admin.', 'Obrero': 'blue-collar', 'Emprendedor': 'entrepreneur',
            'Empleada doméstica': 'housemaid', 'Gerente / directivo': 'management',
            'Jubilado': 'retired', 'Independiente': 'self-employed', 'Servicios': 'services',
            'Estudiante': 'student', 'Técnico': 'technician', 'Desempleado': 'unemployed',
            'Desconocido': 'unknown'}
ESTADO_CIVIL = {'Casado(a)': 'married', 'Soltero(a)': 'single', 'Divorciado(a) / viudo(a)': 'divorced'}
EDUCACION = {'Primaria': 'primary', 'Secundaria': 'secondary', 'Universitaria': 'tertiary',
             'Desconocida': 'unknown'}
SI_NO = {'No': 'no', 'Sí': 'yes'}
CONTACTO = {'Celular': 'cellular', 'Teléfono fijo': 'telephone', 'Desconocido': 'unknown'}
MESES = {'Enero': 'jan', 'Febrero': 'feb', 'Marzo': 'mar', 'Abril': 'apr', 'Mayo': 'may',
         'Junio': 'jun', 'Julio': 'jul', 'Agosto': 'aug', 'Septiembre': 'sep',
         'Octubre': 'oct', 'Noviembre': 'nov', 'Diciembre': 'dec'}
RESULTADO_ANTERIOR = {'Nunca estuvo en una campaña anterior': 'unknown', 'Compró': 'success',
                      'No compró': 'failure', 'Otro': 'other'}

col1, col2 = st.columns(2)

with col1:
    st.subheader('Datos del cliente')
    edad = st.number_input('Edad', min_value=18, max_value=100, value=35)
    trabajo = st.selectbox('Trabajo', list(TRABAJOS))
    estado = st.selectbox('Estado civil', list(ESTADO_CIVIL))
    educacion = st.selectbox('Educación', list(EDUCACION), index=1)
    saldo = st.number_input('Saldo promedio en el banco (euros)', value=1000, step=100)
    mora = st.selectbox('¿Tiene créditos en mora?', list(SI_NO))
    vivienda = st.selectbox('¿Tiene crédito de vivienda?', list(SI_NO))
    prestamo = st.selectbox('¿Tiene crédito personal?', list(SI_NO))

with col2:
    st.subheader('Datos de la campaña')
    contacto = st.selectbox('Medio de contacto', list(CONTACTO))
    mes = st.selectbox('Mes de la llamada', list(MESES), index=4)
    dia = st.number_input('Día del mes de la llamada', min_value=1, max_value=31, value=15)
    llamadas = st.number_input('¿Cuántas veces se le ha llamado en esta campaña? (contando esta)',
                               min_value=1, max_value=60, value=1)
    resultado_ant = st.selectbox('Resultado de la campaña anterior', list(RESULTADO_ANTERIOR))
    if RESULTADO_ANTERIOR[resultado_ant] == 'unknown':
        dias_ant, contactos_ant = -1, 0   # -1 = nunca contactado, igual que en el dataset
    else:
        dias_ant = st.number_input('Días desde el último contacto de la campaña anterior',
                                   min_value=0, max_value=900, value=90)
        contactos_ant = st.number_input('Contactos en campañas anteriores',
                                        min_value=1, max_value=50, value=1)

if st.button('Predecir', type='primary'):
    # 1. Armamos una fila con los mismos nombres y valores del dataset original
    cliente = pd.DataFrame([{
        'age': edad, 'job': TRABAJOS[trabajo], 'marital': ESTADO_CIVIL[estado],
        'education': EDUCACION[educacion], 'default': SI_NO[mora], 'balance': saldo,
        'housing': SI_NO[vivienda], 'loan': SI_NO[prestamo], 'contact': CONTACTO[contacto],
        'day': dia, 'month': MESES[mes], 'campaign': llamadas, 'pdays': dias_ant,
        'previous': contactos_ant, 'poutcome': RESULTADO_ANTERIOR[resultado_ant],
    }])

    # 2. Mismo preprocesamiento del entrenamiento: One-Hot y exactamente las mismas columnas
    #    (las categorías que no aplican a este cliente quedan en 0)
    cliente = pd.get_dummies(cliente, dtype=int).reindex(columns=columnas, fill_value=0)

    # 3. Puntaje del modelo. Ojo: por el class_weight='balanced' NO es una probabilidad real,
    #    por eso lo comparamos contra los clientes del conjunto de prueba.
    puntaje = modelo.predict_proba(cliente)[0, 1]

    # ¿En qué porcentaje de arriba de la lista queda? (1% = de los mejores)
    top = (PUNTAJES_TEST >= puntaje).mean() * 100
    # De los clientes de prueba con un puntaje igual o mayor, ¿cuántos compraron de verdad?
    mismos_o_mejores = PUNTAJES_TEST >= puntaje
    tasa_real = COMPRO_TEST[mismos_o_mejores].mean() if mismos_o_mejores.any() else np.nan

    c1, c2 = st.columns(2)
    c1.metric('Lugar en la lista de llamadas', f'Top {max(top, 0.1):.1f} %')
    c2.metric('Compraron de verdad (clientes así o mejores)',
              f'{tasa_real * 100:.1f} %' if not np.isnan(tasa_real) else 'sin datos')

    if top <= 10:
        st.success(f'Prioridad ALTA: está en el 10 % de clientes con más chance. '
                   f'Un cliente al azar compra el {TASA_BASE * 100:.1f} % de las veces.')
    elif top <= 30:
        st.info(f'Prioridad MEDIA: está en el 30 % de arriba de la lista.')
    else:
        st.warning(f'Prioridad BAJA: está en la parte de abajo de la lista.')
    st.caption(f'Puntaje del modelo: {puntaje:.3f} (sirve para ordenar, no es una probabilidad).')

st.caption('Modelo: Random Forest entrenado con el dataset Bank Marketing (UCI). '
           'Proyecto final de Inteligencia Artificial.')
