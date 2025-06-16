from tools_rec import * 


def compute_true_residuals(y, y_hat): 
    return (y - y_hat) ** 2


def compute_smoothed_residuals(y, baseline_y, alg_y): 
    return (
        ((y - baseline_y) ** 2) + 
        (2 * (y - baseline_y) * (alg_y - y)) + # Covariance term 
        ((alg_y - y) ** 2)
    )

def compute_diff(y, y_hat):
    return (y - y_hat)


def compare_residuals_by_axes(df, residuals, axes, mapping=None): 
    
    out_df = pd.DataFrame(columns=['sum', 'n', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    
    for cat in sorted(df[axes].unique()): 
        print(cat)
        filtered_df = df[df[axes] == cat]
        
        descriptive_stats = filtered_df[residuals].describe()

        out_df_row = {
            'sum': filtered_df[residuals].sum(),
            'n': descriptive_stats['count'],
            'mean': descriptive_stats['mean'],
            'std': descriptive_stats['std'],
            'min': descriptive_stats['min'],
            '25%': descriptive_stats['25%'],
            '50%': descriptive_stats['50%'],
            '75%':  descriptive_stats['75%'],
            'max':  descriptive_stats['max']
        }
        print(out_df_row)
        
        if mapping: 
            name = mapping.get(cat, 'unknown_label')
            out_df.loc[name] = out_df_row
        else: 
            out_df.loc[cat] = out_df_row
        
        out_df = out_df.map(lambda x: np.round(x, 4))
        
    return out_df 
        

def lookup_features_from_top_k_residuals(df, feature_eval, residuals, k): 
    df[residuals] = df[residuals].map(lambda x: np.abs(x))
    data = df.sort_values(by=residuals, ascending=False, inplace=False)
    filtered_df = data[feature_eval][:k]

    descriptive_stats = filtered_df.describe()
    descriptive_stats.map(lambda x: np.round(x, 4))
    return descriptive_stats
        
        
        
        
    

        
        
        
    