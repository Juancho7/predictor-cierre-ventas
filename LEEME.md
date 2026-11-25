# Proyecto final IA - Predictor de cierre de ventas

## Archivos
- `Proyecto_Final_Bank_Marketing.ipynb`: todo el proceso (EDA, limpieza, Random Forest, red neuronal, métricas). Ya está ejecutado, con resultados.
- `bank-full.csv`: el dataset (UCI Bank Marketing).
- `app.py`: la página web (Streamlit).
- `modelo_rf.joblib`, `columnas.joblib`, `referencia_test.joblib`: los genera el notebook; la página los necesita.

## Correr la página en tu PC
1. Abrir una terminal en esta carpeta.
2. `pip install -r requirements.txt` (o con uv: `uv pip install -r requirements.txt`)
3. `streamlit run app.py`
4. Se abre en el navegador (http://localhost:8501).

Importante: la versión de scikit-learn con la que se abre el modelo debe ser parecida a la que lo guardó (1.9).
Si sale error al cargar `modelo_rf.joblib`, vuelve a correr el notebook en tu PC o en Colab para regenerarlo.

## Correr el notebook en Colab
Sube `bank-full.csv` a la sesión (o a tu Drive y cambia `RUTA`). Run all.
