import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("Iniciando fase de Clustering...")
    
    # 1. Cargar datos
    df_pca = pd.read_csv('clientes_pca_final.csv')
    df_rfm = pd.read_csv('clientes_rfm_final.csv')
    
    # Seleccionar solo componentes para el clustering
    X = df_pca.drop(columns=['customer_id'])
    
    # 2. Evaluar distintos valores de K
    best_k = 2
    best_score = -1
    best_labels = None
    
    print("\nEvaluando K-Means (k=2, 3, 4):")
    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        counts = np.bincount(labels)
        print(f"K={k} | Silhouette Score: {score:.4f} | Tamaños de cluster: {counts}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"\n>> Mejor K seleccionado de forma automática: {best_k} (Silhouette: {best_score:.4f})")
    
    # 3. Asignar cluster
    df_pca['Cluster'] = best_labels
    df_rfm['Cluster'] = best_labels
    
    # 4. Construir base final
    cols_to_merge = ['customer_id'] + [c for c in df_pca.columns if c.startswith('PC_')]
    df_final = pd.merge(df_rfm, df_pca[cols_to_merge], on='customer_id', how='left')
    
    # Mover Cluster a una posicion mas visible (indice 1, despues del id)
    cols = list(df_final.columns)
    cols.insert(1, cols.pop(cols.index('Cluster')))
    df_final = df_final[cols]
    
    # 5. Exportar a Excel
    output_excel = 'Segmentacion_Clientes_Resultados.xlsx'
    df_final.to_excel(output_excel, index=False)
    print(f"\nArchivo exportado exitosamente: {output_excel}")
    
    # 6. Análisis Descriptivo e Interpretación
    # Configurar estética ejecutiva
    plt.style.use('seaborn-v0_8-whitegrid')
    executive_palette = ['#1a3a5a', '#6c7a89', '#8b0000', '#005b96']
    sns.set_palette(executive_palette)
    
    os.makedirs('graficos', exist_ok=True)
    
    # a) Visualización PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x='PC_1', y='PC_2', hue='Cluster', palette=executive_palette[:best_k], s=150, alpha=0.9, edgecolor='black')
    plt.title('Distribución de Clusters en Componentes Principales 1 y 2', fontsize=14, fontweight='bold', color='#333333')
    plt.savefig('graficos/pca_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # b) Boxplots de RFM
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(data=df_final, x='Cluster', y='recency', ax=axes[0], palette=executive_palette[:best_k])
    axes[0].set_title('Recency (Días)', fontweight='bold')
    sns.boxplot(data=df_final, x='Cluster', y='frequency', ax=axes[1], palette=executive_palette[:best_k])
    axes[1].set_title('Frequency (Transacciones)', fontweight='bold')
    sns.boxplot(data=df_final, x='Cluster', y='monetary', ax=axes[2], palette=executive_palette[:best_k])
    axes[2].set_title('Monetary (Ingresos)', fontweight='bold')
    plt.suptitle('Análisis RFM por Cluster', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graficos/rfm_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # c) Perfiles (Género por Cluster)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_final, x='Cluster', hue='gender', palette=['#1a3a5a', '#8b0000'])
    plt.title('Distribución de Género por Cluster', fontsize=14, fontweight='bold')
    plt.savefig('graficos/perfiles_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n--- Promedios por Cluster (Interpretación de Negocio) ---")
    cluster_means = df_final.groupby('Cluster')[['recency', 'frequency', 'monetary', 'income_monthly_usd', 'avg_days']].mean()
    print(cluster_means.round(2))
    
    print("\nGráficos generados correctamente en la carpeta 'graficos/'.")
    print("Proceso de Clustering y Visualización finalizado con éxito.")

if __name__ == "__main__":
    main()
