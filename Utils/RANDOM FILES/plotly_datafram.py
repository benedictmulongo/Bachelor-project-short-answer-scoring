import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six
# plotly_datafram
df = pd.DataFrame()
# 
# df[''] = [0,1,2,3,'avg / total']
# df['precision'] = [0.82, 0.36, 0.46, 0.52, 0.53]
# df['recall'] = [0.44, 0.63, 0.46, 0.30,0.47]
# df['f1-score'] = [0.57,0.46, 0.46, 0.38, 0.47]
# df['support'] = [172,220,280,166,841]


# 0.1 U 0.1 L 0.7 TEst 0.1 val iteration = 2
# df[''] = ['Machine learning','Self-learning-rule-based']
# df['Test accuracy'] = [0.50,0.475]
# df['Cross validation'] = [0.529,0.537]

# 0.1 U 0.2 L 0.6 TEst 0.1 val iteration = 4
df[''] = ['Machine learning','Self-learning-rule-based']
df['Test accuracy'] = [0.53,0.50]
df['Cross validation'] = [0.51,0.61]

# df[''] = ['Predicted class 0','Predicted class 1']
# df['True class 0'] = ['TP','FP']
# df['True class 1'] = ['FN','TP']

# df[''] = ['True class 0','True class 1']
# df['Predicted class 0'] = ['TP','FP']
# df['Predicted class 1'] = ['FN','TP']

# Second accuracy =  0.4513888888888889
# second Cross accuracy =  0.5315865118850194
# Length of self_learning =  337
# Data only  Second all Cross accuracy =  0.6665775401069519
# All accuracy =  0.503968253968254
# All Cross accuracy =  0.5102913907284768
# Length of self_learning =  499
# Data only all Cross accuracy =  0.4828163265306122

# Second accuracy =  0.5011185682326622
# second Cross accuracy =  0.6111725663716814
# 
# All accuracy =  0.5302013422818792
# All Cross accuracy =  0.5189796882380727


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    plt.show()
    return ax

render_mpl_table(df, header_columns=0, col_width=4.0)


# top=0.945,
# bottom=0.029,
# left=0.007,
# right=0.989,
# hspace=0.2,
# wspace=0.2