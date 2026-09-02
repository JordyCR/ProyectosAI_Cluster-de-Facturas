import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    print("Cargando dataset...")
    df = pd.read_csv('clientes_rfm_final.csv')
    
    # 1. Feature Engineering de perfil
    print("Generando embeddings numéricos de perfil (usando TF-IDF)...")
    df['perfil_text'] = df['gender'].astype(str) + " " + df['occupation'].astype(str)
    
    # Usar TfidfVectorizer como embedding estadístico local (evita errores de conexión a HuggingFace)
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(df['perfil_text'].tolist()).toarray()
    
    # Crear un dataframe con los embeddings
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols)
    
    # 2. Construcción del dataset analítico
    print("Construyendo dataset analítico...")
    # Seleccionar numéricas
    num_cols = ['income_monthly_usd', 'frequency', 'recency', 'monetary', 'avg_days', 'std_days', 'flag_single_purchase']
    df_num = df[num_cols].copy()
    
    # Integrar
    df_analytical = pd.concat([df_num, df_emb], axis=1)
    
    # 3. Escalado y normalización
    print("Aplicando StandardScaler...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_analytical)
    df_scaled = pd.DataFrame(scaled_data, columns=df_analytical.columns)
    
    print("\nValidación post-escalado (Media ~ 0, Varianza ~ 1):")
    # Mostramos los primeros 5 campos de las estadísticas como muestra
    print("Media de las primeras 5 variables:")
    print(df_scaled.mean().head())
    print("\nVarianza de las primeras 5 variables:")
    print(df_scaled.var().head())
    
    # 4. Reducción de dimensionalidad (PCA)
    print("\nAplicando PCA...")
    pca_full = PCA()
    pca_full.fit(df_scaled)
    
    # Calcular varianza acumulada
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Encontrar número de componentes para explicar al menos 80% (0.80)
    # np.argmax devuelve el primer índice donde la condición es True
    k = np.argmax(cumsum_variance >= 0.80) + 1
    varianza_obtenida = cumsum_variance[k-1]
    print(f"Número de componentes necesarios para explicar >= 80% varianza: {k}")
    print(f"Varianza explicada acumulada con {k} componentes: {varianza_obtenida:.2%}")
    
    # Aplicar PCA con k componentes
    pca_final = PCA(n_components=k)
    reduced_data = pca_final.fit_transform(df_scaled)
    
    # Crear dataset final
    pca_cols = [f'PC_{i+1}' for i in range(k)]
    df_pca = pd.DataFrame(reduced_data, columns=pca_cols)
    
    # Anexar customer_id para trazabilidad
    df_pca.insert(0, 'customer_id', df['customer_id'])
    
    # Guardar
    output_file = 'clientes_pca_final.csv'
    df_pca.to_csv(output_file, index=False)
    
    print(f"\nProceso completado. Dataset reducido guardado en: {output_file}")
    print(f"Dimensiones del dataset PCA final: {df_pca.shape}")

if __name__ == "__main__":
    main()
