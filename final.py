import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
except:
    plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style('whitegrid')
try:
    plt.style.use('seaborn-v0_8-pastel')  
except:
    try:
        plt.style.use('seaborn-pastel')
    except:
        print("无法设置seaborn样式，将使用默认样式")

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Species'] = df['Species'].apply(lambda x: x.replace('Iris-', ''))
    return df

def plot_species_count(df):
    plt.figure(figsize=(10, 6))
    species_counts = df['Species'].value_counts().reset_index()
    species_counts.columns = ['Species', 'Count']
    colors = ['#8884d8', '#82ca9d', '#ffc658']
    try:
        sns.barplot(x='Species', y='Count', data=species_counts, hue='Species', palette=colors, legend=False)
    except:
        sns.barplot(x='Species', y='Count', data=species_counts, palette=colors)
    
    plt.title('Species Count Distribution', fontsize=15)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, count in enumerate(species_counts['Count']):
        plt.text(i, count + 1, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('species_count.png', dpi=300)
    plt.close()

def plot_petal_scatter(df):
    plt.figure(figsize=(10, 8))
    colors = {'setosa': '#8884d8', 'versicolor': '#82ca9d', 'virginica': '#ffc658'}
    
    for species, group in df.groupby('Species'):
        plt.scatter(
            x=group['PetalLengthCm'], 
            y=group['PetalWidthCm'],
            label=species,
            color=colors[species],
            s=70,
            alpha=0.7,
            edgecolor='w'
        )
    
    plt.title('Petal Length vs Width', fontsize=15)
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.legend(title='Species', fontsize=10)
    plt.grid(linestyle='--', alpha=0.7)

    plt.annotate('Setosa cluster', 
                xy=(1.5, 0.3), 
                xytext=(2.5, 0.3),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.savefig('petal_scatter.png', dpi=300)
    plt.close()

def plot_sepal_boxplot(df):
    plt.figure(figsize=(12, 7))
    colors = ['#8884d8', '#82ca9d', '#ffc658']
    bp = sns.boxplot(x='Species', y='SepalLengthCm', data=df, palette=colors)

    sns.stripplot(x='Species', y='SepalLengthCm', data=df, 
                 size=4, color='black', alpha=0.3)
    
    plt.title('Sepal Length Distribution by Species', fontsize=15)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Sepal Length (cm)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, species in enumerate(df['Species'].unique()):
        median = df[df['Species'] == species]['SepalLengthCm'].median()
        plt.text(i, median + 0.1, f'Median: {median}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('sepal_boxplot.png', dpi=300)
    plt.close()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        annot=True,
        fmt='.2f',
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .8},
        mask=mask
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()

def main():
    file_path = 'Iris.csv'
    df = load_data(file_path)
    """
    print("Dataset Info:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData Summary:")
    print(df.describe())

    print("\nStatistics by Species:")
    for species, group in df.groupby('Species'):
        print(f"\n{species}:")
        print(group.describe().loc[['mean', 'std', 'min', 'max']])
    """

    print("\nCreating visualizations...")
    plot_species_count(df)
    plot_petal_scatter(df)
    plot_sepal_boxplot(df)
    plot_correlation_heatmap(df)
    
    print("\nAll charts have been saved to the current directory.")
    
if __name__ == "__main__":
    main()