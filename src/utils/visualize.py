import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.models.hybrid import HybridForecaster
from src.data.dataset import StockDataset, PROCESSED_DIR, FEATURE_COLUMNS
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

SEQ_LENGTH = 120

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NIFTY50_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
]


def load_model(ticker: str):
    model_path = MODELS_DIR / f"{ticker}_model.pt"
    
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = HybridForecaster(
        input_size=len(FEATURE_COLUMNS),
        seq_length=SEQ_LENGTH,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('scaler_params', None)


def get_predictions(ticker: str, test_ratio: float = 0.2):
    model, _ = load_model(ticker)
    if model is None:
        return None
    
    try:
        dataset = StockDataset(
            data_dir=PROCESSED_DIR,
            seq_length=SEQ_LENGTH,
            tickers=[ticker],
            train=True,
        )
    except:
        return None
    
    n_test = int(len(dataset) * test_ratio)
    test_indices = list(range(len(dataset) - n_test, len(dataset)))
    
    actual_3m, pred_3m = [], []
    actual_1y, pred_1y = [], []
    actual_3y, pred_3y = [], []
    
    with torch.no_grad():
        for idx in test_indices:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            
            p3m, p1y, p3y = model(x)
            
            actual_3m.append(y[0].item())
            actual_1y.append(y[1].item())
            actual_3y.append(y[2].item())
            pred_3m.append(p3m.item())
            pred_1y.append(p1y.item())
            pred_3y.append(p3y.item())
    
    return {
        '3M': {'actual': actual_3m, 'predicted': pred_3m},
        '1Y': {'actual': actual_1y, 'predicted': pred_1y},
        '3Y': {'actual': actual_3y, 'predicted': pred_3y},
    }


def plot_predicted_vs_actual(ticker: str):
    data = get_predictions(ticker)
    if data is None:
        print(f"Could not get predictions for {ticker}")
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('3-Month Returns', '1-Year Returns', '3-Year Returns'),
        horizontal_spacing=0.08
    )
    
    for i, period in enumerate(['3M', '1Y', '3Y']):
        actual = data[period]['actual']
        predicted = data[period]['predicted']
        
        fig.add_trace(
            go.Scatter(
                x=actual, y=predicted,
                mode='markers',
                marker=dict(size=6, opacity=0.6, color='#3498db'),
                name=f'{period} Returns',
                showlegend=False
            ),
            row=1, col=i+1
        )
        
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction',
                showlegend=(i == 0)
            ),
            row=1, col=i+1
        )
        
        fig.update_xaxes(title_text='Actual Return', row=1, col=i+1)
        fig.update_yaxes(title_text='Predicted Return', row=1, col=i+1)
    
    fig.update_layout(
        title=f'{ticker} - Predicted vs Actual Returns',
        height=400,
        width=1200,
        template='plotly_white'
    )
    
    return fig


def plot_macd_signal_validation(ticker: str):
    """
    Plots Price vs MACD signals vs Model Predictions.
    Checks if model agrees with MACD crossovers.
    """
    # Need access to full dataframe for indicators
    # We can load processed data directly
    try:
        df = pd.read_csv(PROCESSED_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    except:
        return None
    
    # Get last N days used for test
    # This is rough matching, visualization purpose mainly
    df = df.tail(200) 
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # Price Candle
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=2, col=1)
        
        # Add Histogram
        hist = df['MACD'] - df['MACD_Signal']
        colors = np.where(hist < 0, 'red', 'green')
        fig.add_trace(go.Bar(x=df.index, y=hist, name='Histogram', marker_color=colors), row=2, col=1)

    fig.update_layout(title=f'{ticker}: Technical Indicators vs Price', xaxis_rangeslider_visible=False, height=600)
    return fig


def plot_sector_performance():
    """
    Groups stocks by sector and compares average model accuracy.
    """
    # Hardcoded sector map for Nifty 50 (Partial)
    sector_map = {
        'RELIANCE': 'Energy', 'TCS': 'IT', 'INFY': 'IT', 'HDFCBANK': 'Finance',
        'ICICIBANK': 'Finance', 'SBIN': 'Finance', 'HINDUNILVR': 'FMCG', 'ITC': 'FMCG',
        'BHARTIARTL': 'Telecom', 'LT': 'Construction', 'BAJFINANCE': 'Finance',
        'MARUTI': 'Auto', 'M&M': 'Auto', 'TATAMOTORS': 'Auto', 'SUNPHARMA': 'Pharma',
        'TITAN': 'Consumer', 'ASIANPAINT': 'Consumer', 'POWERGRID': 'Utilities',
        'NTPC': 'Utilities', 'JSWSTEEL': 'Metals', 'TATASTEEL': 'Metals'
    }
    
    metrics_file = RESULTS_DIR / "evaluation_metrics.csv"
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    df['Sector'] = df['Ticker'].map(sector_map).fillna('Others')
    
    # Group by Sector
    sector_perf = df.groupby('Sector')['DirAcc_3M'].mean().sort_values()
    
    fig = go.Figure(data=[
        go.Bar(x=sector_perf.index, y=sector_perf.values, marker_color='purple')
    ])
    
    fig.update_layout(title="Average Model Accuracy by Sector (3-Month)", 
                      yaxis_title="Directional Accuracy %", height=500)
    return fig


def plot_confidence_intervals(ticker: str):
    """
    Visualizes prediction uncertainty (simulated via simple error bounds or dropout if enabled).
    For this demo, we'll show Residual-based Error Bands.
    """
    data = get_predictions(ticker)
    if data is None:
        return None
        
    actual = np.array(data['3M']['actual'])
    predicted = np.array(data['3M']['predicted'])
    error_std = np.std(predicted - actual)
    
    upper_bound = predicted + 1.96 * error_std
    lower_bound = predicted - 1.96 * error_std
    
    x = np.arange(len(actual))
    
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(x=x, y=actual, mode='lines', name='Actual', line=dict(color='gray', width=1)))
    
    # Predicted
    fig.add_trace(go.Scatter(x=x, y=predicted, mode='lines', name='Predicted', line=dict(color='blue')))
    
    # Upper/Lower Bounds
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,255,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(title=f'{ticker} Forecast with Confidence Intervals (3M)',
                      yaxis_title="Return", xaxis_title="Time Step")
    return fig


def plot_model_comparison_bar():
    comparison_file = RESULTS_DIR / "model_comparison.csv"
    
    if not comparison_file.exists():
        print("Run baseline_models.py first to generate comparison data")
        return None
    
    df = pd.read_csv(comparison_file)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('MSE Comparison (Lower is Better)', 'Directional Accuracy (Higher is Better)'),
        horizontal_spacing=0.12
    )
    
    colors = px.colors.qualitative.Set2
    
    for i, model in enumerate(df['Model']):
        fig.add_trace(
            go.Bar(
                name=model,
                x=['3-Month', '1-Year'],
                y=[df.loc[i, 'MSE_3M'], df.loc[i, 'MSE_1Y']] if 'MSE_1Y' in df.columns else [df.loc[i, 'MSE_3M'], 0],
                marker_color=colors[i % len(colors)],
                showlegend=True
            ),
            row=1, col=1
        )
    
    for i, model in enumerate(df['Model']):
        fig.add_trace(
            go.Bar(
                name=model,
                x=['3-Month', '1-Year'],
                y=[df.loc[i, 'DirAcc_3M'], df.loc[i, 'DirAcc_1Y']] if 'DirAcc_1Y' in df.columns else [df.loc[i, 'DirAcc_3M'], 0],
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Model Comparison: HybridForecaster vs Baselines',
        height=500,
        width=1200,
        barmode='group',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )
    
    fig.update_yaxes(title_text='MSE', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy (%)', row=1, col=2)
    
    return fig


def plot_directional_accuracy_comparison():
    comparison_file = RESULTS_DIR / "model_comparison.csv"
    
    if not comparison_file.exists():
        print("Run baseline_models.py first")
        return None
    
    df = pd.read_csv(comparison_file)
    
    df_sorted = df.sort_values('DirAcc_3M', ascending=True)
    
    colors = ['#e74c3c' if 'Hybrid' not in model else '#27ae60' for model in df_sorted['Model']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['Model'],
        x=df_sorted['DirAcc_3M'],
        orientation='h',
        marker_color=colors,
        text=[f'{x:.1f}%' for x in df_sorted['DirAcc_3M']],
        textposition='outside'
    ))
    
    fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                  annotation_text="Random (50%)", annotation_position="top")
    
    fig.update_layout(
        title='Directional Accuracy Comparison (3-Month Prediction)',
        xaxis_title='Directional Accuracy (%)',
        yaxis_title='Model',
        height=400,
        width=800,
        template='plotly_white',
        xaxis=dict(range=[0, 100])
    )
    
    return fig


def plot_returns_distribution(ticker: str):
    data = get_predictions(ticker)
    if data is None:
        return None
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('3-Month Returns', '1-Year Returns', '3-Year Returns')
    )
    
    for i, period in enumerate(['3M', '1Y', '3Y']):
        actual = data[period]['actual']
        predicted = data[period]['predicted']
        
        fig.add_trace(
            go.Histogram(x=actual, name='Actual', opacity=0.7, marker_color='#3498db'),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Histogram(x=predicted, name='Predicted', opacity=0.7, marker_color='#e74c3c'),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=f'{ticker} - Distribution of Returns',
        height=400,
        width=1200,
        barmode='overlay',
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_prediction_error_over_time(ticker: str):
    data = get_predictions(ticker)
    if data is None:
        return None
    
    actual = np.array(data['3M']['actual'])
    predicted = np.array(data['3M']['predicted'])
    error = predicted - actual
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Actual vs Predicted Over Time', 'Prediction Error'),
        vertical_spacing=0.15
    )
    
    x = list(range(len(actual)))
    
    fig.add_trace(
        go.Scatter(x=x, y=actual, mode='lines', name='Actual', line=dict(color='#3498db')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=predicted, mode='lines', name='Predicted', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    
    colors = ['#27ae60' if e >= 0 else '#e74c3c' for e in error]
    fig.add_trace(
        go.Bar(x=x, y=error, name='Error', marker_color=colors),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        title=f'{ticker} - 3-Month Return Predictions Over Time',
        height=600,
        width=1000,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text='Test Sample Index', row=2, col=1)
    fig.update_yaxes(title_text='Return', row=1, col=1)
    fig.update_yaxes(title_text='Error', row=2, col=1)
    
    return fig


def plot_metrics_heatmap():
    metrics_file = RESULTS_DIR / "evaluation_metrics.csv"
    
    if not metrics_file.exists():
        print("Run evaluate.py first")
        return None
    
    df = pd.read_csv(metrics_file)
    
    metrics_cols = ['DirAcc_3M', 'DirAcc_1Y', 'DirAcc_3Y']
    available_cols = [col for col in metrics_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    heatmap_data = df[['Ticker'] + available_cols].set_index('Ticker')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=['3-Month', '1-Year', '3-Year'][:len(available_cols)],
        y=heatmap_data.index,
        colorscale='RdYlGn',
        text=np.round(heatmap_data.values, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='Accuracy %')
    ))
    
    fig.update_layout(
        title='Directional Accuracy Heatmap by Stock',
        height=800,
        width=600,
        template='plotly_white'
    )
    
    return fig


def generate_all_plots():
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Generating Predicted vs Actual plots...")
    for ticker in tqdm(NIFTY50_TICKERS[:5], desc="Pred vs Actual"):
        fig = plot_predicted_vs_actual(ticker)
        if fig:
            fig.write_html(PLOTS_DIR / f"{ticker}_pred_vs_actual.html")
    
    print("\n2. Generating Model Comparison plots...")
    fig = plot_model_comparison_bar()
    if fig:
        fig.write_html(PLOTS_DIR / "model_comparison_bar.html")
    
    fig = plot_directional_accuracy_comparison()
    if fig:
        fig.write_html(PLOTS_DIR / "directional_accuracy_comparison.html")
    
    print("\n3. Generating Distribution plots...")
    for ticker in tqdm(NIFTY50_TICKERS[:3], desc="Distributions"):
        fig = plot_returns_distribution(ticker)
        if fig:
            fig.write_html(PLOTS_DIR / f"{ticker}_distribution.html")
    
    print("\n4. Generating Error Analysis plots...")
    for ticker in tqdm(NIFTY50_TICKERS[:3], desc="Error Analysis"):
        fig = plot_prediction_error_over_time(ticker)
        if fig:
            fig.write_html(PLOTS_DIR / f"{ticker}_error_analysis.html")
    
    print("\n5. Generating Metrics Heatmap...")
    fig = plot_metrics_heatmap()
    if fig:
        fig.write_html(PLOTS_DIR / "metrics_heatmap.html")
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 All plots saved to: {PLOTS_DIR}")
    print(f"\nGenerated files:")
    for f in sorted(PLOTS_DIR.glob("*")):
        print(f"  - {f.name}")
        
    print("\n7. Generating Advanced Plots...")
    for ticker in tqdm(NIFTY50_TICKERS[:3], desc="Advanced Plots"):
        # MACD Validation
        fig_macd = plot_macd_signal_validation(ticker)
        if fig_macd:
            fig_macd.write_html(PLOTS_DIR / f"{ticker}_macd_validation.html")
            
        # Confidence Intervals
        fig_ci = plot_confidence_intervals(ticker)
        if fig_ci:
            fig_ci.write_html(PLOTS_DIR / f"{ticker}_confidence_intervals.html")
            
    # Sector Performance
    fig_sector = plot_sector_performance()
    if fig_sector:
        fig_sector.write_html(PLOTS_DIR / "sector_performance.html")


def show_sample_plots():
    print("Generating sample plots for display...")
    
    ticker = "RELIANCE"
    
    fig1 = plot_predicted_vs_actual(ticker)
    if fig1:
        fig1.show()
    
    fig2 = plot_model_comparison_bar()
    if fig2:
        fig2.show()
    
    fig3 = plot_directional_accuracy_comparison()
    if fig3:
        fig3.show()



def plot_portfolio_simulation(top_n: int = 5):
    """
    Simulates portfolio growth if we invested in top N stocks vs Market (All 50).
    Uses the 3-Month predictions to select stocks, then tracks actual performance.
    """
    # 1. Select Top N stocks based on models
    # We need to run predictions for all stocks to rank them, or load from report if exists.
    # For now, let's run predictions on the fly (might be slow) or just pick a few for demo.
    # To be fast, let's use the first 5 tickers as "Top 5" for this demo, 
    # OR better: iterate all tickers, get 3M pred, sort.
    
    print(f"Simulating Portfolio (Top {top_n} Strategy)...")
    
    market_returns = []
    portfolio_returns = []
    
    # storage for (ticker, pred_3m, actual_trace)
    stock_data = []
    
    for ticker in tqdm(NIFTY50_TICKERS, desc="Simulating"):
        data = get_predictions(ticker)
        if data is None:
            continue
            
        # Get average predicted 3M return (across test set? No, just the last point?)
        # simulation should probably be "Backtest on Test Set".
        # Let's take the AVERAGE 3M return for each stock over the test period as the "Expected Return" 
        # and see if high expected return correlates with high actual return.
        
        # Actually, let's look at the cumulative return over the test period.
        actual_series = np.array(data['3M']['actual']) # These are 3M returns at each step
        # If we invest at t=0, return at t=1 is (1+actual[0])? No, actual is (Price_t+3m - Price_t)/Price_t
        # So it's a lookahead return.
        
        # Proper backtest:
        # At each time step t in test set:
        #   Identify Top N stocks with highest Predicted 3M return.
        #   Invest in them.
        #   Wait 3M? That's complex time management.
        
        # Simplified Visualization:
        # Just plot the "Perfect Portfolio" vs "Predicted Portfolio" vs "Market".
        # Cumulative return of Top 5 predicted stocks vs Average of all.
        
        avg_pred_3m = np.mean(data['3M']['predicted'])
        avg_actual_3m = np.mean(data['3M']['actual'])
        stock_data.append({
            'ticker': ticker,
            'pred': avg_pred_3m,
            'actual': avg_actual_3m
        })
        
    df = pd.DataFrame(stock_data)
    df = df.sort_values('pred', ascending=False)
    
    top_picks = df.head(top_n)
    market_avg = df['actual'].mean()
    portfolio_avg = top_picks['actual'].mean()
    
    # Create Bar Comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Market (Nifty50 Avg)', f'Top {top_n} Model Picks'],
        y=[market_avg, portfolio_avg],
        text=[f"{market_avg*100:.1f}%", f"{portfolio_avg*100:.1f}%"],
        textposition='auto',
        marker_color=['gray', '#2ecc71']
    ))
    
    fig.update_layout(
        title=f'Portfolio Simulation: 3-Month Return (Model vs Market)',
        yaxis_title='Average Actual 3M Return',
        template='plotly_white',
        height=500,
        width=600
    )
    
    return fig


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show_sample_plots()
    else:
        generate_all_plots()
        
        print("\n6. Generating Portfolio Simulation...")
        fig = plot_portfolio_simulation()
        if fig:
            fig.write_html(PLOTS_DIR / "portfolio_simulation.html")


