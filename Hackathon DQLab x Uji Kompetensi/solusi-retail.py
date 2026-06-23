import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

df = pd.read_excel('data_penjualan.xlsx')

daily_df = (
    df.groupby(
        ['tgl_transaksi', 'kode_produk', 'nama_produk'],
        as_index=False
    )['total_nilai']
    .sum()
)

daily_df = daily_df.sort_values(
    by=['kode_produk', 'tgl_transaksi']
)

daily_df['MA_3'] = (
    daily_df
    .groupby('kode_produk')['total_nilai']
    .transform(
        lambda x: x.rolling(
        window=3,
        min_periods=3
        ).mean()
    )
)

daily_df = daily_df.dropna(subset=['MA_3'])

daily_df['Prev_MA'] = (
    daily_df
    .groupby('kode_produk')['MA_3']
    .shift(1)
)

daily_df['Is_Uptrend'] = (
    daily_df['MA_3'] >
    daily_df['Prev_MA']
)

def calculate_streak(series):

    streak = 0
    result = []

    for value in series:

        if value:
            streak += 1
        else:
            streak = 0

        result.append(streak)

    return result

daily_df['Consecutive_Days'] = (
    daily_df
    .groupby('kode_produk')['Is_Uptrend']
    .transform(calculate_streak)
)

final_report = []

for kode_produk, group in daily_df.groupby('kode_produk'):

    if group['Consecutive_Days'].max() < 12:
        continue

    valid_growths = []

    group_reset = group.reset_index(drop=True)

    for streak_len in group_reset['Consecutive_Days'].unique():

        if streak_len < 12:
            continue

        streak_rows = group_reset[
            group_reset['Consecutive_Days'] == streak_len
        ]

        for idx in streak_rows.index:

            start_idx = idx - streak_len + 1

            trend_session = group_reset.iloc[
                start_idx:idx + 1
            ]

            start_value = trend_session['MA_3'].iloc[0]
            end_value = trend_session['MA_3'].iloc[-1]

            if start_value <= 0:
                continue

            growth_pct = (
                (end_value / start_value) - 1
            ) * 100

            valid_growths.append({
                'growth': growth_pct,
                'start': start_value,
                'end': end_value
            })

    if not valid_growths:
        continue

    best_growth = max(
        valid_growths,
        key=lambda x: x['growth']
    )

    final_report.append({

        'kode_produk':
            kode_produk,

        'nama_produk':
            group['nama_produk'].iloc[0],

        'MA_Awal_Tren':
            round(best_growth['start'], 2),

        'MA_Akhir_Tren':
            round(best_growth['end'], 2),

        'Growth_Pct':
            round(best_growth['growth'], 2),

        'Max_Consecutive_Days':
            group['Consecutive_Days'].max()
    })

final_report = pd.DataFrame(final_report)

if final_report.empty:

    final_report = pd.DataFrame(
        columns=[
            'kode_produk',
            'nama_produk',
            'MA_Awal_Tren',
            'MA_Akhir_Tren',
            'Growth_Pct',
            'Max_Consecutive_Days'
        ]
    )

else:

    total_sales_map = (
        df.groupby('kode_produk')['total_nilai']
        .sum()
    )

    final_report['Total Penjualan'] = (
        final_report['kode_produk']
        .map(total_sales_map)
    )

    final_report = final_report.sort_values(
        by='Total Penjualan',
        ascending=False
    )

    final_report = final_report.head(3)

rising_codes = final_report['kode_produk'].unique()

plot_df = daily_df[
    daily_df['kode_produk'].isin(rising_codes)
].copy()

if not plot_df.empty:

    plot_df['Normalized'] = (
        plot_df
        .groupby('kode_produk')['MA_3']
        .transform(
            lambda x: (x / x.iloc[0]) * 100
        )
    )

top3_sales = (
    df.groupby(
        ['kode_produk', 'nama_produk']
    )['total_nilai']
    .sum()
    .reset_index()
    .sort_values(
        by='total_nilai',
        ascending=False
    )
    .head(3)
)

top3_codes = top3_sales['kode_produk'].tolist()

top3_plot_df = daily_df[
    daily_df['kode_produk'].isin(top3_codes)
].copy()

top3_plot_df['Normalized'] = (
    top3_plot_df
    .groupby('kode_produk')['MA_3']
    .transform(
        lambda x: (x / x.iloc[0]) * 100
    )
)

basket_df = df.pivot_table(
    index='nomor_struk',
    columns='nama_produk',
    values='total_nilai',
    aggfunc='count',
    fill_value=0
)

basket_df = (
    basket_df > 0
).astype(int)

basket_df.columns = basket_df.columns.astype(str)

frequent_itemsets = apriori(
    basket_df,
    min_support=0.01,
    use_colnames=True
)

if not frequent_itemsets.empty:

    rules = association_rules(
        frequent_itemsets,
        metric='lift',
        min_threshold=1
    )

else:

    rules = pd.DataFrame(
        columns=[
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]
    )

if not rules.empty:

    rules = rules[
        rules['lift'] >= 2
    ]

    rising_star_names = (
        final_report['nama_produk']
        .unique()
        .tolist()
    )

    def contains_rising_star(row):

        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])

        combined_items = antecedents + consequents

        return any(
            item in rising_star_names
            for item in combined_items
        )

    rules = rules[
        rules.apply(
            contains_rising_star,
            axis=1
        )
    ]

    rules = rules.sort_values(
        by=[
            'lift',
            'support',
            'confidence'
        ],
        ascending=[False, False, False]
    )

    rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

fig = plt.figure(figsize=(15, 8), dpi=100)
ax = fig.add_subplot(111)

sorted_report = final_report.sort_values(
    by='Total Penjualan',
    ascending=False
)

custom_palette = [
    '#FFD700',
    '#C0C0C0',
    '#CD7F32',
    '#2ecc71',
    '#3498db',
    '#9b59b6',
    '#e74c3c',
    '#34495e',
]

default_color = '#95a5a6'

color_mapping = {}
rank_mapping = {}

for i, row in enumerate(sorted_report.itertuples()):

    kode_produk = row.kode_produk

    color_mapping[kode_produk] = (
        custom_palette[i]
        if i < len(custom_palette)
        else default_color
    )

    rank_mapping[kode_produk] = i + 1

grey_colors = [
    '#B0B0B0',
    '#909090',
    '#707070'
]

for idx, (kode_produk, group) in enumerate(
    top3_plot_df.groupby('kode_produk')
):

    nama_produk = group['nama_produk'].iloc[0]

    grey_color = (
        grey_colors[idx]
        if idx < len(grey_colors)
        else '#808080'
    )

    ax.plot(
        group['tgl_transaksi'],
        group['Normalized'],
        linestyle='--',
        linewidth=2,
        marker='o',
        markersize=3,
        color=grey_color,
        alpha=0.7,
        label=f"Top Sales: {nama_produk}"
    )

if not plot_df.empty:

    ranked_codes = final_report['kode_produk'].tolist()

    for kode_produk in ranked_codes:

        group = plot_df[
            plot_df['kode_produk'] == kode_produk
        ]

        nama_produk = group['nama_produk'].iloc[0]

        line_color = color_mapping.get(
            kode_produk,
            default_color
        )

        rank = rank_mapping.get(
            kode_produk,
            '?'
        )

        growth = round(
        final_report[
            final_report['kode_produk'] == kode_produk
        ]['Growth_Pct'].iloc[0],
        2
        )

        label_with_rank = (
        f"Rank {rank}: "
        f"{nama_produk}"
        )

        ax.plot(
            group['tgl_transaksi'],
            group['Normalized'],
            marker='o',
            markersize=4,
            linewidth=2.5,
            color=line_color,
            label=label_with_rank
        )

font_title = {
    'family': 'sans-serif',
    'color': 'black',
    'weight': 'bold',
    'size': 16
}

font_label = {
    'family': 'sans-serif',
    'weight': 'normal',
    'size': 12
}

ax.set_title(
    'ANALISIS PERTUMBUHAN RELATIF PRODUK RISING STAR\n'
    '(Dengan Benchmark Top 3 Total Penjualan)',
    fontdict=font_title,
    pad=20
)

ax.set_xlabel(
    'Periode Tanggal',
    fontdict=font_label,
    labelpad=10
)

ax.set_ylabel(
    'Indeks Pertumbuhan (Base 100)',
    fontdict=font_label,
    labelpad=10
)

ax.grid(
    True,
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

ax.axhline(
    y=100,
    color='black',
    linestyle='-',
    linewidth=1,
    alpha=0.5
)

plt.xticks(
    rotation=45,
    ha='right',
    fontsize=10
)

plt.yticks(fontsize=10)

handles, labels = ax.get_legend_handles_labels()

top_sales_items = []
rising_items = []

for h, l in zip(handles, labels):

    if l.startswith('Top Sales'):
        top_sales_items.append((h, l))
    else:
        rising_items.append((h, l))

if rising_items:

    rising_items = sorted(
        rising_items,
        key=lambda x: int(
            x[1].split(':')[0].split()[1]
        )
    )

final_legend = top_sales_items + rising_items

final_handles = [x[0] for x in final_legend]
final_labels = [x[1] for x in final_legend]

ax.legend(
    final_handles,
    final_labels,
    title="Kategori Produk",
    title_fontsize=12,
    fontsize=10,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=True,
    shadow=True
)

plt.tight_layout()

plt.savefig(
    'rising_star_index.png',
    bbox_inches='tight'
)

fig2 = plt.figure(figsize=(15, 8), dpi=100)
ax2 = fig2.add_subplot(111)

for idx, (kode_produk, group) in enumerate(
    top3_plot_df.groupby('kode_produk')
):

    nama_produk = group['nama_produk'].iloc[0]

    grey_color = (
        grey_colors[idx]
        if idx < len(grey_colors)
        else '#808080'
    )

    ax2.plot(
        group['tgl_transaksi'],
        group['total_nilai'],
        linestyle='--',
        linewidth=2,
        marker='o',
        markersize=3,
        color=grey_color,
        alpha=0.7,
        label=f"Top Sales: {nama_produk}"
    )

if not plot_df.empty:

    ranked_codes = final_report['kode_produk'].tolist()

    for kode_produk in ranked_codes:

        group = plot_df[
            plot_df['kode_produk'] == kode_produk
        ]

        nama_produk = group['nama_produk'].iloc[0]

        line_color = color_mapping.get(
            kode_produk,
            default_color
        )

        rank = rank_mapping.get(
            kode_produk,
            '?'
        )

        label_with_rank = f"Rank {rank}: {nama_produk}"

        ax2.plot(
            group['tgl_transaksi'],
            group['total_nilai'],
            marker='o',
            markersize=4,
            linewidth=2.5,
            color=line_color,
            label=label_with_rank
        )

ax2.set_title(
    'ANALISIS NILAI PENJUALAN PRODUK RISING STAR\n'
    '(Nilai Penjualan Asli)',
    fontdict=font_title,
    pad=20
)

ax2.set_xlabel(
    'Periode Tanggal',
    fontdict=font_label,
    labelpad=10
)

ax2.set_ylabel(
    'Total Nilai Penjualan',
    fontdict=font_label,
    labelpad=10
)

ax2.grid(
    True,
    linestyle='--',
    linewidth=0.5,
    alpha=0.5
)

plt.xticks(
    rotation=45,
    ha='right',
    fontsize=10
)

plt.yticks(fontsize=10)

handles2, labels2 = ax2.get_legend_handles_labels()

top_sales_items2 = []
rising_items2 = []

for h, l in zip(handles2, labels2):

    if l.startswith('Top Sales'):
        top_sales_items2.append((h, l))
    else:
        rising_items2.append((h, l))

if rising_items2:

    rising_items2 = sorted(
        rising_items2,
        key=lambda x: int(
            x[1].split(':')[0].split()[1]
        )
    )

final_legend2 = top_sales_items2 + rising_items2

final_handles2 = [x[0] for x in final_legend2]
final_labels2 = [x[1] for x in final_legend2]

ax2.legend(
    final_handles2,
    final_labels2,
    title="Kategori Produk",
    title_fontsize=12,
    fontsize=10,
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0,
    frameon=True,
    shadow=True
)

plt.tight_layout()

plt.savefig(
    'rising_star_actual.png',
    bbox_inches='tight'
)

rising_star_export = final_report.copy()

total_sales_map = df.groupby('kode_produk')['total_nilai'].sum()

rising_star_export['Total Penjualan'] = rising_star_export['kode_produk'].map(total_sales_map)

rising_star_export = rising_star_export[[
    'kode_produk',
    'nama_produk',
    'Growth_Pct',
    'Total Penjualan'
]]

rising_star_export.columns = [
    'Kode Produk',
    'Nama Produk',
    'Growth %',
    'Total Penjualan'
]

if not rules.empty:

    rules = rules[rules['lift'] >= 2]

    rising_names = set(final_report['nama_produk'])

    def is_rising(row):
        items = list(row['antecedents']) + list(row['consequents'])
        return any(x in rising_names for x in items)

    rules = rules[rules.apply(is_rising, axis=1)]

    def clean(x):
        return ', '.join(sorted(list(x))) if isinstance(x, (set, frozenset)) else str(x)

    rules_export = rules.copy()
    rules_export['Jika Membeli'] = rules_export['antecedents'].apply(clean)
    rules_export['Maka Membeli'] = rules_export['consequents'].apply(clean)
    total_tx = df['nomor_struk'].nunique()
    rules_export['Jumlah Invoice'] = (rules_export['support'] * total_tx).round().astype(int)
    rules_export['Support'] = rules_export['support'].round(2)
    rules_export['Confidence'] = rules_export['confidence'].round(2)
    rules_export['Lift'] = rules_export['lift'].round(2)

    rules_export = rules_export[[
        'Jika Membeli',
        'Maka Membeli',
        'Jumlah Invoice',
        'Support',
        'Confidence',
        'Lift'
    ]]

else:

    rules_export = pd.DataFrame(columns=[
        'Jika Membeli',
        'Maka Membeli',
        'Jumlah Invoice',
        'Support',
        'Confidence',
        'Lift'
    ])

with pd.ExcelWriter(
    'retail_insight.xlsx',
    engine='openpyxl'
) as writer:

    rising_star_export.to_excel(
        writer,
        sheet_name='Rising Star',
        index=False
    )

    rules_export.to_excel(
        writer,
        sheet_name='Potential Packaging',
        index=False
    )
print("\nSemua file berhasil dibuat!")

plt.close(fig)
plt.close(fig2)