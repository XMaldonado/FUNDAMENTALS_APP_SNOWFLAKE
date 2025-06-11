import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go


@st.cache_data
def load_fundamental_data():
    session = get_active_session()
    conn = session.connection
    query = """SELECT DISTINCT
            s.issuer_name,
            --s.SAA_TAXONOMY_SAA_TAXONOMY_LEVEL_2,
            m.ROOT_CUSIP,
            m.SP_COMPANY_NAME,
            m.SP_CUSIP,
            --m.SP_ENTITY_ID,
            --m.SP_ENTITY_ID_1,
            --m.SP_ENTITY_ID_2,
            m.SP_EXCHANGE_TICKER,
            m.SP_ISIN,
            --m.SP_TICKER,
            m.TICKER,
            m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_2,
            m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_3,
            m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4,
            m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_5,
            s.AMOUNT_OUTSTANDING_M,
            m.IQ_CASH_OPER,
            m.IQ_CASH_ST_INVEST,
            m.IQ_DEBT_TO_CAPITAL,
            m.IQ_DEBT_TO_EBITDA,
            m.IQ_EBITDA,
            m.IQ_EBITDA_MARGIN,
            m.IQ_EBIT_INT,
            m.IQ_NI_MARGIN,
            m.IQ_NPPE,
            m.IQ_OPER_INC,
            m.IQ_OPER_INC_1,
            m.IQ_OPER_INC_10,
            m.IQ_OPER_INC_11,
            m.IQ_OPER_INC_12,
            m.IQ_OPER_INC_2,
            m.IQ_OPER_INC_3,
            m.IQ_OPER_INC_4,
            m.IQ_OPER_INC_5,
            m.IQ_OPER_INC_6,
            m.IQ_OPER_INC_7,
            m.IQ_OPER_INC_8,
            m.IQ_OPER_INC_9,
            m.IQ_REV,
            m.IQ_REV_1,
            m.IQ_REV_10,
            m.IQ_REV_11,
            m.IQ_REV_12,
            m.IQ_REV_2,
            m.IQ_REV_3,
            m.IQ_REV_4,
            m.IQ_REV_5,
            m.IQ_REV_6,
            m.IQ_REV_7,
            m.IQ_REV_8,
            m.IQ_REV_9,
            m.IQ_ROA,
            m.IQ_TEV,
            m.IQ_TOTAL_DEBT,
            m.IQ_TOTAL_REV,
            m.IQ_TOTAL_REV_5YR_ANN_GROWTH,
            m.IQ_UFCF_MARGIN,
            m.IQ_UNLEVERED_FCF,
            m.SNL_TEV,
            m.IQ_TOTAL_ASSETS,
            m.RANK
    
        FROM ALADDIN.PROD.SECURITY s
        JOIN TRADERS.CAPITAL_IQ.MAP_MASTER m
            ON SUBSTR(s.PRICING_CUSIP, 1, 6) = m.ROOT_CUSIP
        WHERE
            s.SAA_TAXONOMY_SAA_TAXONOMY_LEVEL_2 != 'Preferred Equity'
            AND s.SAA_TAXONOMY_SAA_TAXONOMY_LEVEL_2 != 'High-Yield Corporate Debt'
            AND m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_1 = 'Corporates'
            AND m.IQ_REV != 0
            AND m.IQ_REV_1 IS NOT NULL
            AND m.EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4 != '0'"""  # Replace with your actual SQL query
    df = pd.read_sql(query, conn)
    return df.drop_duplicates()

# --- Rename Columns ---
def rename_columns(df):
    return df.rename(columns={
        'IQ_OPER_INC': 'Operating Income',
        'IQ_EBITDA': 'EBITDA',
        'IQ_EBITDA_MARGIN': 'EBITDA Margin',
        'IQ_NI_MARGIN': 'Net Income Margin',
        'IQ_ROA': 'Return on Assets',
        'IQ_UFCF_MARGIN': 'FCF Margin',
        'IQ_UNLEVERED_FCF': 'FCF',
        'IQ_TOTAL_DEBT': 'Total Debt',
        'IQ_EBIT_INT': 'Interest Coverage',
        'IQ_DEBT_TO_EBITDA': 'Debt to EBITDA',
        'IQ_TOTAL_REV_5YR_ANN_GROWTH': '5-Yr Revenue CAGR',
        'IQ_CASH_ST_INVEST': 'Total Cash and Investments',
        'IQ_TOTAL_REV': 'Revenue',
        'IQ_CASH_OPER': 'Cashflow From Operations',
        'IQ_DEBT_TO_CAPITAL': 'Debt to Capital',
        'IQ_TOTAL_ASSETS': 'Total Assets',
        'IQ_NPPE': 'PP&E'
    })

# --- Sector Mapping ---
sector_mapping = {
    'Banking': 'Banking', 
    'Basic Industry': 'Basic Industry',
    'Capital Goods': 'Capital Goods',
    'Industrial Other': 'Capital Goods',
    'Communications': 'Communications', 
    'Consumer Cyclical': 'Cyclicals',
    'Energy': 'Energy',
    'Finance Companies': 'Financial',
    'Brokerage/Asset Managers/Exchanges': 'Financial',
    'Financial Other': 'Financial',
    'Insurance': 'Insurance',
    'Consumer Non-Cyclical': 'Non Cyclicals', 
    'REITs': 'REITs',
    'Technology': 'Technology',
    'Transportation': 'Transportation',
    'Natural Gas': 'Utilities',
    'Electric': 'Utilities',
    'Utility Other': 'Utilities'
}

# --- Similarity Function ---

def find_similar_tickers(df, base_ticker, features, top_n=5, weights=None):
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    
    if weights:
        weight_array = np.array([weights.get(f, 1.0) for f in features])
        scaled *= weight_array

    distances = pairwise_distances(scaled, metric='manhattan')
    base_index = df[df['TICKER'] == base_ticker].index[0]
    scores = pd.Series(distances[base_index], index=df.index).drop(base_index).sort_values()
    return df.loc[scores.head(top_n).index][['TICKER', *features]]



def find_similar_tickers1(df, base_ticker, features, top_n=5):
    df = df.dropna(subset=features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    distances = pairwise_distances(scaled, metric='manhattan')
    base_index = df[df['TICKER'] == base_ticker].index[0]
    scores = pd.Series(distances[base_index], index=df.index).drop(base_index).sort_values()
    return df.loc[scores.head(top_n).index][['TICKER', *features]]
    
# --- Radar Chart ---
def plot_radar_comparison2(base_row, peer_avg, metrics, label,top_t):
    for i in range(len(peer_avg)):
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=base_row[metrics].values.flatten().tolist() + [base_row[metrics].values.flatten()[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f'{label}'
        ))
        fig.add_trace(go.Scatterpolar(
            r=peer_avg.iloc[i][metrics].values.flatten().tolist() + [peer_avg.iloc[i][metrics].values.flatten()[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f'{top_t.iloc[i]}'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f'{label} vs {top_t.iloc[i]}'
        )
        st.plotly_chart(fig)
        
def plot_radar_comparison3(base_row, peer_avg, metrics, label,top_t):
    fig = go.Figure()
    base_values = base_row[metrics].values.flatten().tolist()
    fig.add_trace(go.Scatterpolar(
        r=base_values + [base_values[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name=f'{label}'
    ))
    for i in range(len(peer_avg)):
        peer_values = peer_avg.iloc[i][metrics].values.flatten().tolist()
        fig.add_trace(go.Scatterpolar(
            r=peer_values + [peer_values[0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f'{top_t.iloc[i]}')
                     )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f'{label} vs Peers'
    )
    st.plotly_chart(fig)

page = st.sidebar.selectbox("Choose Page", ["Home", "IG","HY"])

if page == "Home":
    st.title("Welcome!")
    st.markdown("This is the home page.")
    #st.image("City Scape.jpg")


elif page == "IG":
    st.title("Investment Grade Dashboard")

    df = load_fundamental_data()
    df = rename_columns(df)
    df['Sector'] = df['EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4'].map(sector_mapping)

    sector_options = sorted(df['Sector'].dropna().unique())
    selected_sector = st.selectbox("Select Sector", sector_options)

    filtered_df = df[df['Sector'] == selected_sector].sort_values(by=['TICKER', 'ROOT_CUSIP'])

    st.subheader(f"{selected_sector} Sector Data")
    st.dataframe(filtered_df[['ROOT_CUSIP', 'TICKER', 'ISSUER_NAME', 'EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4','Revenue', 'EBITDA',
                                   'EBITDA Margin',
                                   'Net Income Margin',
                                   'Return on Assets',
                                   'FCF Margin',
                                   'FCF',
                                   'Total Debt',
                                   'Interest Coverage',
                                   'Debt to EBITDA',
                                   '5-Yr Revenue CAGR',
                                   'Total Cash and Investments',
                                   'Operating Income']].drop_duplicates().reset_index(drop=True), use_container_width=True)

    if selected_sector == 'Technology':
        numeric_cols = ['Revenue',
                        'EBITDA',
                       'EBITDA Margin',
                       'Net Income Margin',
                       'Return on Assets',
                       'FCF Margin',
                        'FCF',
                        'Total Debt',
                        'Interest Coverage',
                        'Debt to EBITDA',
                        '5-Yr Revenue CAGR',
                        'Total Cash and Investments',
                        'Operating Income']
        tech_df = filtered_df[numeric_cols].drop_duplicates().apply(pd.to_numeric, errors='coerce')
        avg_df = pd.DataFrame([tech_df.mean()])
        st.subheader("Average Metrics")
        st.dataframe(avg_df, use_container_width=True)

        ticker_input = st.text_input("Enter a TICKER to find similar peers")
        if ticker_input and ticker_input.upper() in df['TICKER'].values:
            filtered_df1 = filtered_df[['TICKER','Revenue', 'EBITDA',
                                   'EBITDA Margin',
                                   'Net Income Margin',
                                   'Return on Assets',
                                   'FCF Margin',
                                   'FCF',
                                   'Total Debt',
                                   'Interest Coverage',
                                   'Debt to EBITDA',
                                   '5-Yr Revenue CAGR',
                                   'Total Cash and Investments',
                                   'Operating Income']].drop_duplicates()

            filtered_df1.replace('NM', 0, inplace=True)
            filtered_df1 = filtered_df1.reset_index(drop=True)
            weights = {}
            for feature in numeric_cols:
                weights[feature] = st.number_input(f"Weight for {feature}", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            similar_df = find_similar_tickers(filtered_df1.drop_duplicates(), ticker_input.upper(), features=numeric_cols,weights=weights)
            #st.subheader(f"{ticker_input.upper()}")
            base_row = filtered_df1[filtered_df1['TICKER'] == ticker_input.upper()]
            st.write(base_row)
            st.subheader("Similar Tickers")
            similar_df = similar_df.reset_index(drop=True)
            st.dataframe(similar_df.drop_duplicates())
            peer_avg = similar_df[numeric_cols]
            base_row = base_row.reset_index(drop=True)
            plot_radar_comparison3(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])
            plot_radar_comparison2(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])


    if selected_sector == 'Communications':
        numeric_cols = ['5-Yr Revenue CAGR',
                                   'EBITDA',
                                   'EBITDA Margin',
                                   'Total Debt',
                                   'FCF Margin',
                                   'Debt to EBITDA',
                                   'Debt to Capital',
                                   'Interest Coverage',
                                   'Total Cash and Investments',
                                   'Operating Income',
                                   'FCF',
                                   'Return on Assets'
                       ]
        tech_df = filtered_df[numeric_cols].drop_duplicates().apply(pd.to_numeric, errors='coerce')
        avg_df = pd.DataFrame([tech_df.mean()])
        st.subheader("Average Metrics")
        st.dataframe(avg_df, use_container_width=True)

        ticker_input = st.text_input("Enter a TICKER to find similar peers")
        if ticker_input and ticker_input.upper() in df['TICKER'].values:
            filtered_df1 = filtered_df[['TICKER','5-Yr Revenue CAGR',
                                   'EBITDA',
                                   'EBITDA Margin',
                                   'Total Debt',
                                   'FCF Margin',
                                   'Debt to EBITDA',
                                   'Debt to Capital',
                                   'Interest Coverage',
                                   'Total Cash and Investments',
                                   'Operating Income',
                                   'FCF',
                                   'Return on Assets'
                                       ]].drop_duplicates()

            filtered_df1.replace('NM', 0, inplace=True)
            filtered_df1 = filtered_df1.reset_index(drop=True)
            
            st.subheader("Assign Weights to Features")
            weights = {}
            for feature in numeric_cols:
                weights[feature] = st.number_input(f"Weight for {feature}", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            similar_df = find_similar_tickers(filtered_df1.drop_duplicates(), ticker_input.upper(), features=numeric_cols,weights=weights)
            #st.subheader(f"{ticker_input.upper()}")
            base_row = filtered_df1[filtered_df1['TICKER'] == ticker_input.upper()]
            base_row = base_row.reset_index(drop=True)
            st.write(base_row)
            st.subheader("Similar Tickers")
            similar_df = similar_df.reset_index(drop=True)
            st.dataframe(similar_df.drop_duplicates())
            peer_avg = similar_df[numeric_cols]
            #st.dataframe(peer_avg)
            #st.write(similar_df['TICKER'])
            plot_radar_comparison3(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])
            plot_radar_comparison2(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])

elif page == "HY":
    st.title("High Yield")

    df = load_fundamental_data()
    df = rename_columns(df)
    df['Sector'] = df['EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4'].map(sector_mapping)

    sector_options = sorted(df['Sector'].dropna().unique())
    selected_sector = st.selectbox("Select Sector", sector_options)

    filtered_df = df[df['Sector'] == selected_sector].sort_values(by=['TICKER', 'ROOT_CUSIP'])

    st.subheader(f"{selected_sector} Sector Data")
    st.dataframe(filtered_df[['ROOT_CUSIP', 'TICKER', 'ISSUER_NAME', 'EXTENDED_CORE_BREAKOUT_SAGE_CORE_LEVEL_4','Revenue', 'EBITDA',
                                   'EBITDA Margin',
                                   'Net Income Margin',
                                   'Return on Assets',
                                   'FCF Margin',
                                   'FCF',
                                   'Total Debt',
                                   'Interest Coverage',
                                   'Debt to EBITDA',
                                   '5-Yr Revenue CAGR',
                                   'Total Cash and Investments',
                                   'Operating Income']].drop_duplicates().reset_index(drop=True), use_container_width=True)

    if selected_sector == 'Technology':
        numeric_cols = ['Revenue',
                        'EBITDA',
                       'EBITDA Margin',
                       'Net Income Margin',
                       'Return on Assets',
                       'FCF Margin',
                        'FCF',
                        'Total Debt',
                        'Interest Coverage',
                        'Debt to EBITDA',
                        '5-Yr Revenue CAGR',
                        'Total Cash and Investments',
                        'Operating Income']
        tech_df = filtered_df[numeric_cols].drop_duplicates().apply(pd.to_numeric, errors='coerce')
        avg_df = pd.DataFrame([tech_df.mean()])
        st.subheader("Average Metrics")
        st.dataframe(avg_df, use_container_width=True)

        ticker_input = st.text_input("Enter a TICKER to find similar peers")
        if ticker_input and ticker_input.upper() in df['TICKER'].values:
            filtered_df1 = filtered_df[['TICKER','Revenue', 'EBITDA',
                                   'EBITDA Margin',
                                   'Net Income Margin',
                                   'Return on Assets',
                                   'FCF Margin',
                                   'FCF',
                                   'Total Debt',
                                   'Interest Coverage',
                                   'Debt to EBITDA',
                                   '5-Yr Revenue CAGR',
                                   'Total Cash and Investments',
                                   'Operating Income']].drop_duplicates()

            filtered_df1.replace('NM', 0, inplace=True)
            filtered_df1 = filtered_df1.reset_index(drop=True)
            similar_df = find_similar_tickers1(filtered_df1.drop_duplicates(), ticker_input.upper(), features=numeric_cols)
            #st.subheader(f"{ticker_input.upper()}")
            base_row = filtered_df1[filtered_df1['TICKER'] == ticker_input.upper()]
            st.write(base_row)
            st.subheader("Similar Tickers")
            similar_df = similar_df.reset_index(drop=True)
            st.dataframe(similar_df.drop_duplicates())
            peer_avg = similar_df[numeric_cols]
            base_row = base_row.reset_index(drop=True)
            plot_radar_comparison3(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])
            plot_radar_comparison2(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])


    if selected_sector == 'Communications':
        numeric_cols = [
                                   
                                   'EBITDA Margin',
                                   'FCF Margin',
                                   'Debt to EBITDA',
                                   'Debt to Capital',
                                   'Return on Assets'
                       ]
        tech_df = filtered_df[numeric_cols].drop_duplicates().apply(pd.to_numeric, errors='coerce')
        avg_df = pd.DataFrame([tech_df.mean()])
        st.subheader("Average Metrics")
        st.dataframe(avg_df, use_container_width=True)

        ticker_input = st.text_input("Enter a TICKER to find similar peers")
        if ticker_input and ticker_input.upper() in df['TICKER'].values:
            filtered_df1 = filtered_df[['TICKER',
                                   
                                   'EBITDA Margin',
                                   'FCF Margin',
                                   'Debt to EBITDA',
                                   'Debt to Capital',
                                   'Return on Assets'
                                       ]].drop_duplicates()

            filtered_df1.replace('NM', 0, inplace=True)
            filtered_df1 = filtered_df1.reset_index(drop=True)
            
            st.subheader("Assign Weights to Features")
            similar_df = find_similar_tickers1(filtered_df1.drop_duplicates(), ticker_input.upper(), features=numeric_cols)
            base_row = filtered_df1[filtered_df1['TICKER'] == ticker_input.upper()]
            base_row = base_row.reset_index(drop=True)
            st.write(base_row)
            st.subheader("Similar Tickers")
            similar_df = similar_df.reset_index(drop=True)
            st.dataframe(similar_df.drop_duplicates())
            peer_avg = similar_df[numeric_cols]
            plot_radar_comparison3(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])
            plot_radar_comparison2(base_row, peer_avg, numeric_cols, ticker_input.upper(),similar_df['TICKER'])

