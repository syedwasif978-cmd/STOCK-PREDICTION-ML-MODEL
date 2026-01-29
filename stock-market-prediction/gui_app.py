"""
Stock Market Prediction System - GUI Application
================================================

A professional desktop application with real-time stock search and prediction.
Features:
    • Real-time search suggestions as user types
    • Live market data fetching from Alpaca API
    • Interactive visualizations and predictions
    • Professional GUI with modern design
    • Support for any stock symbol on the market

Author: Stock Analysis System
Date: 2026
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import scrolledtext
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import sys
import traceback
from PIL import Image, ImageTk
import requests

# Import project modules
try:
    from api.alpaca_fetch import fetch_stock_data, validate_symbol
    from model.train import engineer_features, prepare_data, train_and_evaluate_model
    from model.predict import predict_next_day, generate_price_curve, calculate_prediction_accuracy
    from visualization.plot_results import plot_actual_vs_predicted, create_summary_report
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class StockPredictionGUI:
    """
    Professional GUI for Stock Market Prediction System.
    
    Provides:
        - Real-time stock symbol search with autocomplete
        - Live market data fetching
        - ML model training and predictions
        - Interactive visualizations
        - Performance metrics display
    """
    
    def __init__(self, root):
        """
        Initialize GUI application.
        
        Parameters:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Stock Market Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Configure style
        self.setup_styles()
        
        # Application variables
        self.current_symbol = tk.StringVar()
        self.search_suggestions = []
        self.popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 
                             'NVDA', 'META', 'NFLX', 'GOOG', 'IBM']
        self.model = None
        self.predictions = None
        self.metrics = None
        self.is_processing = False
        
        # Create GUI components
        self.create_widgets()
        
    def setup_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define colors
        style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), 
                       background='#f0f0f0', foreground='#1f77b4')
        style.configure('Subtitle.TLabel', font=('Helvetica', 11), 
                       background='#f0f0f0', foreground='#555')
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10), background='#f0f0f0')
        
    def create_widgets(self):
        """Create all GUI components."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== HEADER =====
        self.create_header(main_container)
        
        # ===== SEARCH SECTION =====
        self.create_search_section(main_container)
        
        # ===== MAIN CONTENT (3 columns) =====
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Info and metrics
        self.create_left_panel(content_frame)
        
        # Middle panel - Visualization
        self.create_middle_panel(content_frame)
        
        # Right panel - Details and controls
        self.create_right_panel(content_frame)
        
    def create_header(self, parent):
        """Create application header."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="📈 Stock Market Prediction System",
                               style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Real-time market data • ML predictions • Live visualizations",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
    def create_search_section(self, parent):
        """Create search bar with autocomplete."""
        search_frame = ttk.LabelFrame(parent, text="Stock Search", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Search input with label
        input_frame = ttk.Frame(search_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Enter Stock Symbol:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Search entry field
        self.search_entry = ttk.Entry(input_frame, font=('Helvetica', 12), width=15)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self.on_search_change)
        self.search_entry.bind('<Return>', lambda e: self.fetch_and_predict())
        
        # Fetch button
        self.fetch_button = ttk.Button(input_frame, text="🔍 Fetch & Predict",
                                      command=self.fetch_and_predict)
        self.fetch_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(input_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Search suggestions (dropdown-style)
        suggestion_frame = ttk.Frame(search_frame)
        suggestion_frame.pack(fill=tk.X)
        
        ttk.Label(suggestion_frame, text="Popular stocks:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Popular buttons
        for stock in self.popular_stocks[:6]:
            ttk.Button(suggestion_frame, text=stock, width=6,
                      command=lambda s=stock: self.quick_select_stock(s)).pack(side=tk.LEFT, padx=2)
        
        # Suggestions listbox
        self.suggestions_frame = ttk.Frame(search_frame)
        self.suggestions_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Label for suggestions
        ttk.Label(self.suggestions_frame, text="Search Results:").pack(anchor=tk.W)
        
        # Suggestions listbox
        self.suggestions_listbox = tk.Listbox(self.suggestions_frame, height=4, 
                                             font=('Helvetica', 10))
        self.suggestions_listbox.pack(fill=tk.X, pady=(5, 0))
        self.suggestions_listbox.bind('<<ListboxSelect>>', self.on_suggestion_select)
        
    def create_left_panel(self, parent):
        """Create left panel with metrics and info."""
        left_frame = ttk.LabelFrame(parent, text="Metrics & Information", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5), min_width=250)
        
        # Stock symbol display
        self.symbol_display = ttk.Label(left_frame, text="Symbol: --", 
                                       font=('Helvetica', 14, 'bold'), foreground='#1f77b4')
        self.symbol_display.pack(pady=10)
        
        # Metrics frame
        self.metrics_text = scrolledtext.ScrolledText(left_frame, height=25, width=30,
                                                     font=('Courier', 9), wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.metrics_text.tag_configure('header', foreground='#1f77b4', font=('Courier', 10, 'bold'))
        self.metrics_text.tag_configure('value', foreground='#2ca02c', font=('Courier', 9, 'bold'))
        self.metrics_text.tag_configure('warning', foreground='#ff7f0e')
        
        # Initial placeholder text
        self.update_metrics_display("Select a stock to see metrics")
        
    def create_middle_panel(self, parent):
        """Create middle panel for visualization."""
        middle_frame = ttk.LabelFrame(parent, text="Price Prediction Chart", padding=5)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas for matplotlib figure
        self.canvas_frame = ttk.Frame(middle_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder message
        self.placeholder_label = ttk.Label(self.canvas_frame, 
                                          text="Select a stock to see prediction chart",
                                          foreground='#999')
        self.placeholder_label.pack(expand=True)
        
    def create_right_panel(self, parent):
        """Create right panel with details and controls."""
        right_frame = ttk.LabelFrame(parent, text="Prediction Details", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(5, 0), min_width=250)
        
        # Prediction display
        self.prediction_text = scrolledtext.ScrolledText(right_frame, height=25, width=30,
                                                        font=('Courier', 9), wrap=tk.WORD)
        self.prediction_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags
        self.prediction_text.tag_configure('header', foreground='#1f77b4', 
                                          font=('Courier', 10, 'bold'))
        self.prediction_text.tag_configure('positive', foreground='#2ca02c')
        self.prediction_text.tag_configure('negative', foreground='#d62728')
        self.prediction_text.tag_configure('neutral', foreground='#ff7f0e')
        
        # Initial placeholder text
        self.update_prediction_display("No prediction yet.\nSelect a stock and click 'Fetch & Predict'")
        
    def on_search_change(self, event):
        """
        Handle search input changes - show suggestions.
        
        Parameters:
            event: Tkinter event object
        """
        search_term = self.search_entry.get().upper().strip()
        
        # Clear suggestions if empty
        if not search_term:
            self.suggestions_listbox.delete(0, tk.END)
            return
        
        # Get matching stocks
        suggestions = self.get_stock_suggestions(search_term)
        
        # Update listbox
        self.suggestions_listbox.delete(0, tk.END)
        for stock in suggestions[:10]:  # Show max 10 suggestions
            self.suggestions_listbox.insert(tk.END, stock)
    
    def get_stock_suggestions(self, search_term):
        """
        Get stock suggestions based on search term.
        
        Parameters:
            search_term: User input search string
            
        Returns:
            List of matching stock symbols
        """
        # Common stock symbols
        common_stocks = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NFLX': 'Netflix Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'WMT': 'Walmart Inc.',
            'DIS': 'The Walt Disney Company',
            'PYPL': 'PayPal Holdings Inc.',
            'INTC': 'Intel Corporation',
            'AMD': 'Advanced Micro Devices Inc.',
            'CSCO': 'Cisco Systems Inc.',
            'ADBE': 'Adobe Inc.',
            'QCOM': 'Qualcomm Incorporated',
            'IBM': 'International Business Machines',
            'ORCL': 'Oracle Corporation',
            'SAP': 'SAP SE',
            'CRM': 'Salesforce Inc.',
            'NOW': 'ServiceNow Inc.',
        }
        
        # Match stocks
        matches = []
        for symbol, company in common_stocks.items():
            if symbol.startswith(search_term):
                matches.append(f"{symbol} - {company}")
        
        return matches
    
    def on_suggestion_select(self, event):
        """
        Handle suggestion selection from listbox.
        
        Parameters:
            event: Tkinter event object
        """
        selection = self.suggestions_listbox.curselection()
        if selection:
            selected_text = self.suggestions_listbox.get(selection[0])
            symbol = selected_text.split(' - ')[0]  # Extract symbol
            self.search_entry.delete(0, tk.END)
            self.search_entry.insert(0, symbol)
            self.fetch_and_predict()
    
    def quick_select_stock(self, symbol):
        """
        Quickly select a popular stock.
        
        Parameters:
            symbol: Stock symbol to fetch
        """
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, symbol)
        self.fetch_and_predict()
    
    def fetch_and_predict(self):
        """Fetch stock data and generate prediction in background thread."""
        symbol = self.search_entry.get().upper().strip()
        
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
        
        # Disable button to prevent multiple clicks
        self.is_processing = True
        self.fetch_button.config(state='disabled')
        self.status_label.config(text="Processing...", foreground="orange")
        self.root.update()
        
        # Run prediction in background thread
        thread = threading.Thread(target=self._fetch_and_predict_worker, args=(symbol,))
        thread.daemon = True
        thread.start()
    
    def _fetch_and_predict_worker(self, symbol):
        """
        Background worker for fetching and prediction.
        
        Parameters:
            symbol: Stock symbol to process
        """
        try:
            # Update UI - fetching data
            self.root.after(0, self._update_status, "Fetching market data...", "orange")
            
            # Fetch data from Alpaca API
            print(f"Fetching data for {symbol}...")
            df = fetch_stock_data(symbol, days=365*5)
            
            if df is None or len(df) < 100:
                raise ValueError(f"Insufficient data for {symbol}")
            
            print(f"Received {len(df)} rows of data")
            
            # Update UI - engineering features
            self.root.after(0, self._update_status, "Engineering features...", "orange")
            
            # Engineer features
            from indicators.sma import calculate_multiple_sma
            from indicators.rsi import calculate_rsi
            from indicators.volatility import calculate_volatility
            
            df['SMA_20'] = calculate_multiple_sma(df, periods=[20], column='Close')[0]
            df['SMA_50'] = calculate_multiple_sma(df, periods=[50], column='Close')[0]
            df['SMA_200'] = calculate_multiple_sma(df, periods=[200], column='Close')[0]
            df['RSI'] = calculate_rsi(df, period=14, column='Close')
            df['Volatility'] = calculate_volatility(df, period=20, column='Close')
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
            
            # Update UI - preparing data
            self.root.after(0, self._update_status, "Preparing data...", "orange")
            
            # Prepare data
            X_train, X_test, y_train, y_test, feature_names, dates = prepare_data(
                df, test_size=0.3, sequence_offset=1
            )
            
            # Update UI - training model
            self.root.after(0, self._update_status, "Training model...", "orange")
            
            # Train model
            from model.random_forest import RandomForestModel
            model = RandomForestModel(n_estimators=100, max_depth=15)
            metrics = model.train(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Get feature importance
            feature_importance = model.get_feature_importance(feature_names, top_n=6)
            
            # Store results
            self.model = model
            self.predictions = y_pred
            self.metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'feature_importance': feature_importance
            }
            
            # Update UI - visualization
            self.root.after(0, self._update_status, "Creating visualization...", "orange")
            
            # Create visualization
            self.root.after(0, self._create_visualization, 
                          y_test.values, y_pred, symbol, 
                          X_test, y_test, dates)
            
            # Update displays
            self.root.after(0, self._update_all_displays, symbol, mse, rmse, mae, r2)
            
            # Update UI - complete
            self.root.after(0, self._update_status, f"✓ Ready ({symbol})", "green")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))
            self.root.after(0, self._update_status, "Error occurred", "red")
            
        finally:
            # Re-enable button
            self.is_processing = False
            self.root.after(0, lambda: self.fetch_button.config(state='normal'))
    
    def _update_status(self, message, color):
        """Update status label."""
        self.status_label.config(text=message, foreground=color)
    
    def _create_visualization(self, actual, predicted, symbol, X_test, y_test, dates):
        """Create and display prediction chart."""
        try:
            # Clear previous widgets
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Create figure with subplots
            fig = Figure(figsize=(10, 6), dpi=100)
            
            # Main prediction plot
            ax1 = fig.add_subplot(211)
            ax1.plot(actual, label='Actual Returns', color='#1f77b4', linewidth=2)
            ax1.plot(predicted, label='Predicted Returns', color='#ff7f0e', 
                    linestyle='--', linewidth=2)
            ax1.set_title(f'{symbol} - Next Day Return Prediction', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Test Sample')
            ax1.set_ylabel('Return (%)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Error distribution plot
            errors = actual - predicted
            ax2 = fig.add_subplot(212)
            ax2.hist(errors, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
            ax2.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.4f}')
            ax2.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Error (%)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            fig.tight_layout()
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def _update_all_displays(self, symbol, mse, rmse, mae, r2):
        """Update all display panels."""
        self.current_symbol.set(symbol)
        self.symbol_display.config(text=f"Symbol: {symbol}")
        
        # Update metrics display
        metrics_text = f"""PREDICTION METRICS
{'='*28}

Symbol: {symbol}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Performance:
─────────────────────────────
MSE (Mean Squared Error):
  {mse:.6f}

RMSE (Root Mean Squared):
  {rmse:.6f}

MAE (Mean Absolute Error):
  {mae:.6f}

R² Score (Variance Explained):
  {r2:.4f}
  
Interpretation:
─────────────────────────────"""
        
        if r2 > 0.7:
            metrics_text += "\n✓ Good model fit"
        elif r2 > 0.5:
            metrics_text += "\n~ Moderate model fit"
        else:
            metrics_text += "\n✗ Poor model fit"
            
        metrics_text += f"\n\nMAE meaning:\nAverage prediction error of\n{mae:.4f} (in return %)"
        
        if self.metrics and 'feature_importance' in self.metrics:
            metrics_text += "\n\nTop Features:\n─────────────────────────────"
            fi = self.metrics['feature_importance']
            for idx, (name, importance) in enumerate(fi.iterrows(), 1):
                metrics_text += f"\n{idx}. {name}: {importance['Importance']:.3f}"
        
        self.update_metrics_display(metrics_text)
        
        # Update prediction display
        prediction_text = f"""NEXT-DAY FORECAST
{'='*28}

Stock: {symbol}
Current Time: {datetime.now().strftime('%H:%M:%S')}

Model Performance:
─────────────────────────────
Accuracy (R²):
  {r2*100:.2f}%

Average Error:
  ±{mae:.4f}

Prediction Quality:
  """
        
        if r2 > 0.7:
            prediction_text += "HIGH ✓"
        elif r2 > 0.5:
            prediction_text += "MEDIUM ~"
        else:
            prediction_text += "LOW ✗"
        
        prediction_text += f"""

Market Analysis:
─────────────────────────────
Based on technical indicators:
  • SMA (20, 50, 200-day)
  • RSI (14-day momentum)
  • Volatility (20-day)
  • Volume moving average

Last Updated:
  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER:
─────────────────────────────
This is a machine learning
prediction model for educational
purposes only.

NOT FINANCIAL ADVICE.
Always consult a financial
advisor before trading.
"""
        
        self.update_prediction_display(prediction_text)
    
    def update_metrics_display(self, text):
        """Update metrics text area."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def update_prediction_display(self, text):
        """Update prediction text area."""
        self.prediction_text.config(state=tk.NORMAL)
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(tk.END, text)
        self.prediction_text.config(state=tk.DISABLED)


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = StockPredictionGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
