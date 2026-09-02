import pandas as pd
import numpy as np

def main():
    print("Iniciando procesamiento de datos...")
    
    # 1. Ingesta y Validación
    # Cargar tablas
    df_clientes = pd.read_csv('clientes_simulados.csv')
    df_ventas = pd.read_csv('ventas_simuladas.csv')
    
    # Validar tipos de datos
    df_ventas['order_date'] = pd.to_datetime(df_ventas['order_date'], format='%d/%m/%Y')
    
    # Validar nulos y duplicados
    print(f"Duplicados en clientes: {df_clientes.duplicated().sum()}")
    print(f"Duplicados en ventas: {df_ventas.duplicated().sum()}")
    
    df_clientes = df_clientes.drop_duplicates()
    df_ventas = df_ventas.drop_duplicates()
    
    # 2. Preparación de datos
    # Definir período de análisis (fecha máxima + 1 día)
    fecha_referencia = df_ventas['order_date'].max() + pd.Timedelta(days=1)
    print(f"Fecha de referencia para Recency: {fecha_referencia}")
    
    # Ordenar compras
    df_ventas = df_ventas.sort_values(by=['customer_id', 'order_date'], ascending=[True, True])
    
    # 3. Feature Engineering Transaccional
    # Calcular RFM a nivel cliente
    agregaciones = {
        'order_date': 'max',             # Para Recency
        'order_id': 'nunique',           # Para Frequency
        'revenue': 'sum',                # Para Monetary
        'days_since_prev_purchase': ['mean', 'std']  # Métricas temporales
    }
    
    df_rfm = df_ventas.groupby('customer_id').agg(agregaciones).reset_index()
    
    # Aplanar las columnas MultiIndex resultantes de agg
    df_rfm.columns = ['customer_id', 'last_order_date', 'frequency', 'monetary', 'avg_days', 'std_days']
    
    # Calcular Recency
    df_rfm['recency'] = (fecha_referencia - df_rfm['last_order_date']).dt.days
    
    # Manejo de clientes con una sola compra
    # Si Frequency == 1, consideramos que es compra única.
    # Alternativamente, clientes cuya std_days es NaN o avg_days es NaN también califican.
    df_rfm['flag_single_purchase'] = np.where(df_rfm['frequency'] == 1, 1, 0)
    
    # Identificar un valor alto para imputar avg_days (ej. el valor máximo de avg_days)
    valor_alto_avg = df_rfm['avg_days'].max()
    print(f"Valor imputado para avg_days en compras únicas: {valor_alto_avg}")
    
    # Imputar
    df_rfm.loc[df_rfm['flag_single_purchase'] == 1, 'std_days'] = 0.0
    df_rfm.loc[df_rfm['flag_single_purchase'] == 1, 'avg_days'] = valor_alto_avg
    
    # Para clientes con frequency > 1 pero que la std es NaN (ej. todas sus compras fueron el mismo día)
    df_rfm['std_days'] = df_rfm['std_days'].fillna(0)
    df_rfm['avg_days'] = df_rfm['avg_days'].fillna(valor_alto_avg)
    
    # Eliminar columna auxiliar de fecha
    df_rfm = df_rfm.drop(columns=['last_order_date'])
    
    # Unir con la tabla de clientes
    df_final = pd.merge(df_clientes, df_rfm, on='customer_id', how='left')
    
    # Guardar resultado
    output_file = 'clientes_rfm_final.csv'
    df_final.to_csv(output_file, index=False)
    
    print("Procesamiento completado.")
    print(f"Dimensiones del dataset final: {df_final.shape}")
    print(f"Dataset guardado en {output_file}")
    
    print("\nMuestra de los resultados:")
    print(df_final[['customer_id', 'frequency', 'recency', 'monetary', 'avg_days', 'std_days', 'flag_single_purchase']].head())

if __name__ == "__main__":
    main()
