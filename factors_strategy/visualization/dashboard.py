"""
Visualization Dashboard for Stock Strategy
Creates interactive dashboards and reports
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class StrategyDashboard:
    """Interactive dashboard for strategy monitoring"""
    
    def __init__(self, config_path: str = "config/strategy.yaml"):
        """Initialize dashboard"""
        self.config = self._load_config(config_path)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AI Stock Selection Strategy Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total Return", className="card-title"),
                            html.H2(id="total-return", children="0.00%"),
                            html.P("Since inception", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sharpe Ratio", className="card-title"),
                            html.H2(id="sharpe-ratio", children="0.00"),
                            html.P("Risk-adjusted return", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Win Rate", className="card-title"),
                            html.H2(id="win-rate", children="0.00%"),
                            html.P("Profitable trades", className="text-muted")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Positions", className="card-title"),
                            html.H2(id="active-positions", children="0"),
                            html.P("Current holdings", className="text-muted")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="performance-chart")
                ], width=8),
                
                dbc.Col([
                    dcc.Graph(id="sector-allocation")
                ], width=4)
            ], className="mb-4"),
            
            # Factor Performance
            dbc.Row([
                dbc.Col([
                    html.H3("Factor Performance"),
                    dcc.Graph(id="factor-ic-chart")
                ], width=6),
                
                dbc.Col([
                    html.H3("Model Predictions"),
                    dcc.Graph(id="model-accuracy-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Recent Recommendations
            dbc.Row([
                dbc.Col([
                    html.H3("Recent Stock Recommendations"),
                    html.Div(id="recommendations-table")
                ])
            ], className="mb-4"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ], fluid=True)
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('total-return', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('win-rate', 'children'),
             Output('active-positions', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_summary_cards(n):
            """Update summary statistics"""
            # In production, fetch from database
            # This is placeholder data
            return "15.23%", "1.85", "68.5%", "8"
            
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            """Update performance chart"""
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            returns = np.random.randn(252) * 0.02
            cumulative_returns = (1 + returns).cumprod()
            
            fig = go.Figure()
            
            # Portfolio performance
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
            
            # Benchmark (CSI 300)
            benchmark_returns = np.random.randn(252) * 0.015
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_cumulative,
                mode='lines',
                name='CSI 300',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig.update_layout(
                title="Cumulative Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                showlegend=True
            )
            
            return fig
            
        @self.app.callback(
            Output('sector-allocation', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_sector_allocation(n):
            """Update sector allocation pie chart"""
            # Sample data
            sectors = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Industrial']
            values = [30, 25, 20, 15, 10]
            
            fig = go.Figure(data=[go.Pie(
                labels=sectors,
                values=values,
                hole=0.3
            )])
            
            fig.update_layout(
                title="Sector Allocation",
                showlegend=True
            )
            
            return fig
            
        @self.app.callback(
            Output('factor-ic-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_factor_ic_chart(n):
            """Update factor IC chart"""
            # Sample factor data
            factors = ['Order Flow', 'Volatility', 'Momentum', 'Liquidity', 'AI Factor 1']
            ic_values = [0.045, 0.038, 0.032, 0.028, 0.052]
            
            fig = go.Figure(data=[
                go.Bar(x=factors, y=ic_values, marker_color='lightblue')
            ])
            
            fig.update_layout(
                title="Factor Information Coefficients",
                xaxis_title="Factor",
                yaxis_title="IC",
                showlegend=False
            )
            
            return fig
            
        @self.app.callback(
            Output('model-accuracy-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_accuracy(n):
            """Update model accuracy chart"""
            models = ['CNN', 'LSTM', 'XGBoost', 'LightGBM', 'Ensemble']
            accuracy = [0.72, 0.70, 0.75, 0.74, 0.78]
            
            fig = go.Figure(data=[
                go.Bar(x=models, y=accuracy, marker_color='lightgreen')
            ])
            
            fig.update_layout(
                title="Model Prediction Accuracy",
                xaxis_title="Model",
                yaxis_title="Accuracy",
                yaxis_range=[0, 1],
                showlegend=False
            )
            
            return fig
            
        @self.app.callback(
            Output('recommendations-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recommendations_table(n):
            """Update recommendations table"""
            # Sample recommendations data
            data = {
                'Symbol': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ'],
                'Name': ['平安银行', '万科A', '浦发银行', '招商银行', '五粮液'],
                'Score': [0.92, 0.89, 0.87, 0.85, 0.83],
                'Expected Return': ['12.5%', '11.2%', '10.8%', '10.3%', '9.8%'],
                'Position Size': ['15%', '15%', '12%', '10%', '8%']
            }
            
            df = pd.DataFrame(data)
            
            table = dbc.Table.from_dataframe(
                df, 
                striped=True, 
                bordered=True, 
                hover=True,
                responsive=True,
                className="table-sm"
            )
            
            return table
            
    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=port)


def create_static_report(date: datetime, output_path: str = "output/reports"):
    """Create static HTML report"""
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Cumulative Performance', 'Daily Returns Distribution',
                       'Factor Performance', 'Stock Recommendations',
                       'Risk Metrics', 'Model Performance'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "table"}],
               [{"type": "indicator"}, {"type": "bar"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Cumulative Performance
    dates = pd.date_range(end=date, periods=252, freq='D')
    returns = np.random.randn(252) * 0.02
    cumulative_returns = (1 + returns).cumprod()
    
    fig.add_trace(
        go.Scatter(x=dates, y=cumulative_returns, name='Strategy'),
        row=1, col=1
    )
    
    # 2. Returns Distribution
    fig.add_trace(
        go.Histogram(x=returns, nbinsx=50, name='Daily Returns'),
        row=1, col=2
    )
    
    # 3. Factor Performance
    factors = ['Order Flow', 'Volatility', 'Momentum', 'Liquidity']
    ic_values = [0.045, 0.038, 0.032, 0.028]
    
    fig.add_trace(
        go.Bar(x=factors, y=ic_values, name='Factor IC'),
        row=2, col=1
    )
    
    # 4. Stock Recommendations Table
    recommendations = go.Table(
        header=dict(values=['Symbol', 'Score', 'Expected Return'],
                   fill_color='lightgray',
                   align='left'),
        cells=dict(values=[['000001.SZ', '000002.SZ', '600000.SH'],
                          [0.92, 0.89, 0.87],
                          ['12.5%', '11.2%', '10.8%']],
                  fill_color='white',
                  align='left')
    )
    
    fig.add_trace(recommendations, row=2, col=2)
    
    # 5. Risk Metrics
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=1.85,
            title={'text': "Sharpe Ratio"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 3]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 1], 'color': "lightgray"},
                       {'range': [1, 2], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 2}}
        ),
        row=3, col=1
    )
    
    # 6. Model Performance
    models = ['CNN', 'LSTM', 'XGBoost', 'Ensemble']
    accuracy = [0.72, 0.70, 0.75, 0.78]
    
    fig.add_trace(
        go.Bar(x=models, y=accuracy, name='Model Accuracy'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=False,
        title_text=f"Stock Strategy Report - {date.strftime('%Y-%m-%d')}",
        title_font_size=24
    )
    
    # Save report
    report_path = Path(output_path) / f"strategy_report_{date.strftime('%Y%m%d')}.html"
    fig.write_html(str(report_path))
    
    logger.info(f"Report saved to: {report_path}")
    
    return report_path


def create_factor_analysis_plot(factor_data: pd.DataFrame) -> go.Figure:
    """Create factor analysis visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Factor Correlation Matrix', 'Factor IC Over Time',
                       'Factor Turnover', 'Factor Contribution to Returns')
    )
    
    # 1. Correlation Matrix
    corr_matrix = factor_data.corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ),
        row=1, col=1
    )
    
    # 2. IC Over Time (placeholder)
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    for factor in factor_data.columns[:3]:
        ic_values = np.random.randn(60) * 0.01 + 0.03
        fig.add_trace(
            go.Scatter(x=dates, y=ic_values, name=factor),
            row=1, col=2
        )
    
    # 3. Factor Turnover
    turnover_data = factor_data.diff().abs().mean()
    fig.add_trace(
        go.Bar(x=turnover_data.index, y=turnover_data.values),
        row=2, col=1
    )
    
    # 4. Factor Contribution
    contributions = factor_data.mean() * 0.1  # Simplified
    fig.add_trace(
        go.Bar(x=contributions.index, y=contributions.values),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    
    return fig


if __name__ == "__main__":
    # Run dashboard
    dashboard = StrategyDashboard()
    dashboard.run(debug=True)