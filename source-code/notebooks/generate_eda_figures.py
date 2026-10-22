import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def generate_all_eda_figures():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "reports", "figures")
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train.csv")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} training samples.")
    
    # Set overall visual style
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("mako", 5)

    # 1. Label Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    counts = df['Label'].value_counts().sort_index()
    percentages = (counts / len(df)) * 100
    
    bars = ax1.bar(counts.index, counts.values, color=['#e11d48', '#38bdf8', '#0284c7', '#0369a1', '#075985'], edgecolor='black', alpha=0.85)
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.set_title('Frequency of Reading Preference Labels (1 - 5)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Label (1 = Do not read, 2-5 = Reading priority)', fontsize=11)
    ax1.set_ylabel('Paper Count', fontsize=11)
    ax1.set_ylim(0, max(counts.values) + 120)

    # Binary distribution: Label 1 (Unwanted) vs Labels 2-5 (Worth reading)
    binary_counts = [counts[1], counts.loc[2:5].sum()]
    labels = [f'Label 1: Unwanted\n({binary_counts[0]} papers)', f'Labels 2-5: Worth Reading\n({binary_counts[1]} papers)']
    colors = ['#e11d48', '#0284c7']
    ax2.pie(binary_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.06, 0), textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Binary Ratio: Filter Target (Stage 1)', fontsize=13, fontweight='bold', pad=12)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_label_distribution.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # 2. Publication Year vs Label
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    year_counts = df.groupby(['year', 'Label']).size().unstack(fill_value=0)
    year_counts.plot(kind='bar', stacked=True, ax=ax1, colormap='Spectral', edgecolor='black', alpha=0.9)
    ax1.set_title('Publication Year vs Reading Preference', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Publication Year', fontsize=11)
    ax1.set_ylabel('Number of Papers', fontsize=11)
    ax1.legend(title='Label', bbox_to_anchor=(1, 1))

    # Mean rating by year
    mean_by_year = df.groupby('year')['Label'].agg(['mean', 'count']).reset_index()
    ax2.plot(mean_by_year['year'].astype(str), mean_by_year['mean'], marker='o', linewidth=2.5, color='#0f766e')
    for _, row in mean_by_year.iterrows():
        ax2.annotate(f"{row['mean']:.2f}\n(n={int(row['count'])})", (str(int(row['year'])), row['mean']),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    ax2.set_title('Average User Rating by Publication Year', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Average Rating', fontsize=11)
    ax2.set_ylim(1, 5)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_year_vs_label.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # 3. Top Venues Analysis
    plt.figure(figsize=(12, 6))
    top_venues = df['venue'].value_counts().head(12).index
    venue_df = df[df['venue'].isin(top_venues)]
    order = df.groupby('venue')['Label'].mean().loc[top_venues].sort_values(ascending=False).index
    
    sns.boxplot(data=venue_df, x='venue', y='Label', order=order, palette='crest')
    plt.title('Rating Distribution Across Top 12 Publication Venues', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Venue', fontsize=11)
    plt.ylabel('User Rating (1 - 5)', fontsize=11)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_top_venues.png")
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    # 4. Title Length & Word Count
    df['title_len'] = df['title'].fillna('').str.len()
    df['word_count'] = df['title'].fillna('').apply(lambda s: len(s.split()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.violinplot(data=df, x='Label', y='word_count', ax=ax1, palette='Blues')
    ax1.set_title('Title Word Count by Label', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Label', fontsize=11)
    ax1.set_ylabel('Word Count', fontsize=11)

    sns.kdeplot(data=df, x='title_len', hue='Label', common_norm=False, palette='turbo', ax=ax2, linewidth=2)
    ax2.set_title('Title Character Length Density by Label', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Character Count', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "04_title_length_analysis.png")
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Saved: {fig4_path}")

    # 5. Top Keywords Comparison: High Priority (4-5) vs Unwanted (1)
    def extract_keywords(text_series):
        stopwords = set(['a', 'an', 'the', 'in', 'on', 'of', 'for', 'with', 'and', 'to', 'using', 'based', 'from', 'by', 'at', 'proceedings', 'international', 'conference', 'symposium', 'vol', 'pp', 'september', 'october', 'june', 'july', 'august', 'may', 'april', 'march', 'february', 'january', 'annual', 'workshop'])
        words = []
        for text in text_series.dropna():
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend([w for w in tokens if w not in stopwords])
        return Counter(words)

    high_kw = extract_keywords(df[df['Label'] >= 4]['title']).most_common(12)
    low_kw = extract_keywords(df[df['Label'] == 1]['title']).most_common(12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    kws1, counts1 = zip(*high_kw)
    ax1.barh(kws1[::-1], counts1[::-1], color='#059669', edgecolor='black', alpha=0.85)
    ax1.set_title('Top Keywords in High Priority Papers (Labels 4 - 5)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Frequency', fontsize=11)

    kws2, counts2 = zip(*low_kw)
    ax2.barh(kws2[::-1], counts2[::-1], color='#dc2626', edgecolor='black', alpha=0.85)
    ax2.set_title('Top Keywords in Unwanted Papers (Label 1)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Frequency', fontsize=11)

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "05_top_keywords_comparison.png")
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("\nAll 5 EDA figures have been successfully generated!")

if __name__ == "__main__":
    generate_all_eda_figures()
