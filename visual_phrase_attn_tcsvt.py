import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib import font_manager as fm

def draw_phrase_attn_grid(save_path="phrase_attn.pdf"):
    # 数据：每个元素是 (attn_list, label_list)
    data_groups = [
        ([0.0710, 0.1047, 0.3820, 0.1090, 0.3333], ["[sot]", "[horse]", "[left]", "[.]", "[eot]"]),
        ([0.1044, 0.0546, 0.1024, 0.1726, 0.2966, 0.0361, 0.0425, 0.1908], 
         ["[sot]", "[horse]", "[in]", "[center]", "[larger]", "[one]", "[.]", "[EOS]"]),
        ([0.1943, 0.0971, 0.0762, 0.4120, 0.0524, 0.1680], ["[sot]", "[cop]", "[on]", "[right]", "[.]", "[eot]"]),
        ([0.0569, 0.0205, 0.1934, 0.4552, 0.0428, 0.2312], ["[sot]", "[elephant]", "[far]", "[left]", "[.]", "[eot]"]),
    ]

    font_path = "TIMES.TTF"       # 坐标轴、colorbar 字体
    title_font_path = "TIMESBD.TTF"  # 标题粗体字体

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
    axes = axes.flatten()

    for idx, (values, labels) in enumerate(data_groups):
        ax = axes[idx]
        norm = plt.Normalize(min(values), max(values))
        cmap = cm.get_cmap("plasma")
        colors = cmap(norm(values))

        # 标签处理
        def clean_label(l):
            if l.lower() == "[sot]":
                return "[SOS]"
            elif l.lower() in {"[eot]", "[eos]"}:
                return "[EOS]"
            else:
                return l.strip("[]")

        new_labels = [clean_label(l) for l in labels]
        ax.bar(range(len(values)), values, tick_label=new_labels, color=colors)

        # 设置标题
        filtered_labels = [clean_label(l) for l in labels
                           if clean_label(l) not in {"[SOS]", "[EOS]"} and l != "[.]"]
        title_str = " ".join(filtered_labels)
        title_prop = fm.FontProperties(fname=title_font_path, size=30)
        ax.set_title(title_str, fontproperties=title_prop)  # 已是粗体

        # 设置坐标刻度字体 + x轴旋转
        tick_prop = fm.FontProperties(fname=font_path, size=25)
        ax.set_xticklabels(new_labels, fontproperties=tick_prop, rotation=30)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(tick_prop)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # 添加 colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.03)
        cbar_prop = fm.FontProperties(fname=font_path, size=30)
        cbar.set_label("Phrase Attention", fontproperties=cbar_prop)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontproperties(cbar_prop)

    plt.tight_layout(pad=4.0)  # 增加子图间距
    plt.savefig(save_path, format="pdf")
    plt.close()

# 调用
draw_phrase_attn_grid("phrase_attn.pdf")
