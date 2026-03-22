import pandas as pd
import os
import glob

# Configuració
OUTPUT_FILE = 'consolidated_data.csv'

def inspect_csv_files():
    """Inspecciona tots els CSVs al directori per veure la seva estructura"""
    print("="*60)
    print("INSPECCIONANT ARXIUS CSV...")
    print("="*60)
    
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("❌ No s'han trobat arxius CSV")
        return {}
    
    file_info = {}
    
    for file in csv_files:
        print(f"\n📄 {file}")
        try:
            df = pd.read_csv(file, nrows=5)
            total_rows = len(pd.read_csv(file))
            print(f"   Files totals: {total_rows}")
            print(f"   Columnes ({len(df.columns)}):")
            for i, col in enumerate(df.columns[:10], 1):  # Mostrar només primeres 10
                print(f"      {i}. {col}")
            if len(df.columns) > 10:
                print(f"      ... i {len(df.columns) - 10} més")
            
            file_info[file] = {
                'columns': list(df.columns),
                'shape': (total_rows, len(df.columns)),
                'sample': df.head(3)
            }
        except Exception as e:
            print(f"   ⚠️  Error llegint arxiu: {e}")
    
    return file_info

def auto_detect_dataset_config(file_info):
    """Detecta automàticament la configuració dels datasets"""
    
    configs = {}
    
    for filename, info in file_info.items():
        columns = info['columns']
        
        config = {
            'file': filename,
            'format': 'long',
            'key_columns': [],
            'value_columns': []
        }
        
        # Detectar columnes clau comunes
        if 'Entity' in columns:
            config['key_columns'].append('Entity')
        if 'Country' in columns or 'country' in columns:
            country_col = [c for c in columns if c.lower() == 'country'][0]
            config['key_columns'].append(country_col)
        if 'Country Name' in columns:
            config['key_columns'].append('Country Name')
        if 'Code' in columns:
            config['key_columns'].append('Code')
        if 'Country Code' in columns:
            config['key_columns'].append('Country Code')
        if 'iso3c' in columns:
            config['key_columns'].append('iso3c')
        if 'Year' in columns:
            config['key_columns'].append('Year')
        
        # Detectar si és format wide (World Bank)
        year_cols = [col for col in columns if '[YR' in col]
        if year_cols:
            config['format'] = 'wide'
            config['year_columns'] = year_cols
        
        # Detectar columnes HDI (només les que tenen format hdi_YYYY)
        hdi_cols = [col for col in columns if col.startswith('hdi_') and col[4:].isdigit()]
        if hdi_cols:
            config['format'] = 'wide_years'
            config['year_columns'] = hdi_cols
            # Les altres columnes que no són anys
            config['non_year_columns'] = [col for col in columns if col not in hdi_cols]
        
        # Les columnes que no són clau són values
        if config['format'] == 'long':
            value_cols = [col for col in columns if col not in config['key_columns']]
            config['value_columns'] = value_cols
        
        configs[filename] = config
        
    return configs

def read_and_process_dataset(filename, config):
    """Llegeix i processa un dataset segons la seva configuració"""
    print(f"\nProcessant {filename}...")
    
    try:
        df = pd.read_csv(config['file'])
    except FileNotFoundError:
        print(f"⚠️  Arxiu no trobat: {config['file']}")
        return None
    
    # Processar segons el format
    if config['format'] == 'wide':
        # Format World Bank: convertir de wide a long
        year_cols = config['year_columns']
        
        # Identificar columnes ID
        id_vars = [col for col in df.columns if col not in year_cols]
        
        # Convertir a format long
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name='Year',
            value_name='Value'
        )
        
        # Netejar columna Year
        df_long['Year'] = df_long['Year'].str.extract(r'(\d{4})')[0].astype(int)
        
        # Si hi ha Series Code, pivotar
        if 'Series Code' in df_long.columns:
            # Crear pivot amb Country, Code, Year com a index
            index_cols = [col for col in id_vars if col not in ['Series Name', 'Series Code']]
            index_cols.append('Year')
            
            df_pivot = df_long.pivot_table(
                index=index_cols,
                columns='Series Code',
                values='Value',
                aggfunc='first'
            ).reset_index()
            
            # Renombrar amb Series Name
            if 'Series Name' in df.columns:
                series_names = df[['Series Code', 'Series Name']].drop_duplicates()
                column_mapping = dict(zip(series_names['Series Code'], series_names['Series Name']))
                df_pivot = df_pivot.rename(columns=column_mapping)
            
            return df_pivot
        
        return df_long
    
    elif config['format'] == 'wide_years':
        # Format HDI
        year_cols = config['year_columns']
        non_year_cols = config['non_year_columns']
        
        # Fer melt només amb les columnes d'anys
        df_long = df.melt(
            id_vars=non_year_cols,
            value_vars=year_cols,
            var_name='Year',
            value_name='HDI'
        )
        
        # Extreure any (eliminar prefix 'hdi_')
        df_long['Year'] = df_long['Year'].str.replace('hdi_', '').astype(int)
        
        return df_long
    
    else:
        # Format long: retornar tal qual
        return df

def consolidate_datasets(configs):
    """Consolida tots els datasets"""
    
    print("\n" + "="*60)
    print("PROCESSANT I CONSOLIDANT DATASETS")
    print("="*60)
    
    # Processar tots els datasets
    processed_dfs = {}
    for filename, config in configs.items():
        df = read_and_process_dataset(filename, config)
        if df is not None:
            processed_dfs[filename] = df
            print(f"✓ {len(df):,} files, {len(df.columns)} columnes")
    
    if not processed_dfs:
        print("❌ No s'ha pogut processar cap dataset")
        return None
    
    print("\n" + "="*60)
    print("CONSOLIDANT TOTS ELS DATASETS...")
    print("="*60)
    
    # Normalitzar noms de columnes en tots els DataFrames
    normalized_dfs = []
    
    for filename, df in processed_dfs.items():
        df_norm = df.copy()
        
        # Normalitzar noms de columnes clau
        rename_map = {
            'Entity': 'Country',
            'country': 'Country',
            'Country Name': 'Country',
            'Country Code': 'Code',
            'iso3c': 'Code'
        }
        
        df_norm = df_norm.rename(columns=rename_map)
        
        # Afegir prefixe al nom del fitxer per identificar origen
        source_name = filename.replace('.csv', '').replace('-', '_')
        
        normalized_dfs.append({
            'name': source_name,
            'df': df_norm
        })
    
    # Començar amb el primer dataset
    result = normalized_dfs[0]['df'].copy()
    print(f"Base: {normalized_dfs[0]['name']}")
    print(f"  Columnes: {list(result.columns)}")
    
    # Fer merge successius
    for item in normalized_dfs[1:]:
        name = item['name']
        df = item['df']
        
        # Determinar columnes de merge
        merge_cols = []
        for col in ['Country', 'Code', 'Year']:
            if col in result.columns and col in df.columns:
                merge_cols.append(col)
        
        if not merge_cols:
            print(f"⚠️  Saltant {name}: no hi ha columnes compatibles")
            print(f"     Result cols: {list(result.columns)[:5]}")
            print(f"     DF cols: {list(df.columns)[:5]}")
            continue
        
        print(f"\nMerging {name}")
        print(f"  Columnes merge: {merge_cols}")
        print(f"  Files abans: {len(result):,}")
        
        # Fer merge
        result = result.merge(
            df,
            on=merge_cols,
            how='outer',
            suffixes=('', f'_{name}')
        )
        
        print(f"  Files després: {len(result):,}, Columnes: {len(result.columns)}")
    
    return result

def clean_data(df):
    """Neteja el dataset final"""
    print("\n" + "="*60)
    print("NETEJANT DADES...")
    print("="*60)
    
    # Reemplaçar '..' per NaN
    df = df.replace('..', pd.NA)
    df = df.replace('', pd.NA)
    
    # Convertir columnes numèriques
    for col in df.columns:
        if col not in ['Country', 'Code', 'tier_hdi']:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
    
    # Ordenar
    sort_cols = []
    if 'Country' in df.columns:
        sort_cols.append('Country')
    if 'Year' in df.columns:
        sort_cols.append('Year')
    
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    # Eliminar columnes completament buides
    df = df.dropna(axis=1, how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"✓ Dataset final: {len(df):,} files, {len(df.columns)} columnes")
    
    return df

def main():
    """Funció principal"""
    
    # Pas 1: Inspeccionar arxius
    file_info = inspect_csv_files()
    
    if not file_info:
        return
    
    input("\n⏸️  Prem ENTER per continuar amb la consolidació...")
    
    # Pas 2: Detectar configuració automàticament
    configs = auto_detect_dataset_config(file_info)
    
    print("\n" + "="*60)
    print("CONFIGURACIÓ DETECTADA:")
    print("="*60)
    for filename, config in configs.items():
        print(f"\n{filename}:")
        print(f"  Format: {config['format']}")
        print(f"  Claus: {config['key_columns']}")
        if config['format'] == 'wide_years':
            print(f"  Anys: {len(config['year_columns'])} columnes")
        elif 'value_columns' in config and config['value_columns']:
            print(f"  Values: {config['value_columns'][:3]}{'...' if len(config['value_columns']) > 3 else ''}")
    
    input("\n⏸️  Prem ENTER per consolidar...")
    
    # Pas 3: Consolidar
    consolidated_df = consolidate_datasets(configs)
    
    if consolidated_df is None:
        return
    
    # Pas 4: Netejar
    final_df = clean_data(consolidated_df)
    
    # Pas 5: Guardar
    print("\n" + "="*60)
    print(f"GUARDANT A {OUTPUT_FILE}...")
    print("="*60)
    
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    file_size = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
    
    print(f"✓ Arxiu guardat!")
    print(f"\n📊 RESUM FINAL:")
    print(f"  Files: {len(final_df):,}")
    print(f"  Columnes: {len(final_df.columns)}")
    print(f"  Mida: {file_size:.2f} MB")
    
    # Previsualització
    print("\n" + "="*60)
    print("PREVISUALITZACIÓ:")
    print("="*60)
    print(final_df.head(10).to_string(max_colwidth=20))
    
    print("\n" + "="*60)
    print("COLUMNES DISPONIBLES:")
    print("="*60)
    for i, col in enumerate(final_df.columns, 1):
        non_null = final_df[col].notna().sum()
        pct = (non_null / len(final_df)) * 100
        print(f"{i:3d}. {col:60s} ({non_null:>7,} values, {pct:5.1f}%)")
    
    # Mostrar estadístiques de països i anys
    if 'Country' in final_df.columns:
        print(f"\n📍 Països únics: {final_df['Country'].nunique()}")
    if 'Year' in final_df.columns:
        print(f"📅 Rang d'anys: {final_df['Year'].min():.0f} - {final_df['Year'].max():.0f}")

if __name__ == "__main__":
    main()