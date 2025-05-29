"""
LLM-Enhanced Factor Generator
Uses Large Language Models to generate and optimize trading factors
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import yaml
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class GeneratedFactor:
    """Data class for LLM-generated factors"""
    name: str
    formula: str
    description: str
    economic_intuition: str
    required_data: List[str]
    parameters: Dict[str, Any]
    expected_ic: float
    confidence_score: float
    generation_timestamp: datetime
    

class LLMFactorGenerator:
    """Generate trading factors using Large Language Models"""
    
    def __init__(self, config_path: str = "config/strategy.yaml"):
        """Initialize LLM factor generator"""
        self.config = self._load_config(config_path)
        self.llm_config = self.config['llm_config']
        self.generated_factors: List[GeneratedFactor] = []
        self.factor_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load LLM configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['strategy']
        
    async def generate_factors(self, 
                             market_data: pd.DataFrame,
                             existing_factors: pd.DataFrame,
                             performance_metrics: Dict) -> List[GeneratedFactor]:
        """Generate new factors based on market conditions"""
        
        # Analyze current market patterns
        market_analysis = self._analyze_market_patterns(market_data)
        
        # Evaluate existing factor performance
        factor_analysis = self._analyze_factor_performance(existing_factors, performance_metrics)
        
        # Generate new factors using LLM
        new_factors = await self._call_llm_for_factors(market_analysis, factor_analysis)
        
        # Validate and refine factors
        validated_factors = self._validate_factors(new_factors, market_data)
        
        self.generated_factors.extend(validated_factors)
        return validated_factors
        
    def _analyze_market_patterns(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market patterns for LLM context"""
        analysis = {
            'volatility_regime': self._detect_volatility_regime(market_data),
            'trend_strength': self._calculate_trend_strength(market_data),
            'market_microstructure': self._analyze_microstructure(market_data),
            'anomalies': self._detect_anomalies(market_data),
            'sector_rotation': self._analyze_sector_rotation(market_data)
        }
        return analysis
        
    def _detect_volatility_regime(self, data: pd.DataFrame) -> str:
        """Detect current volatility regime"""
        # Calculate realized volatility
        returns = data.groupby('symbol')['price'].pct_change()
        current_vol = returns.rolling('1h').std().mean()
        historical_vol = returns.rolling('24h').std().mean()
        
        if current_vol > historical_vol * 1.5:
            return "high_volatility"
        elif current_vol < historical_vol * 0.7:
            return "low_volatility"
        else:
            return "normal_volatility"
            
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate market trend strength"""
        # Simple trend analysis
        prices = data.groupby('symbol')['price'].last()
        returns_1h = prices.pct_change(periods=3600)  # 1 hour
        returns_1d = prices.pct_change(periods=86400)  # 1 day
        
        return {
            'short_term_momentum': returns_1h.mean(),
            'medium_term_momentum': returns_1d.mean(),
            'trend_consistency': (returns_1h * returns_1d > 0).mean()
        }
        
    def _analyze_microstructure(self, data: pd.DataFrame) -> Dict:
        """Analyze market microstructure patterns"""
        return {
            'avg_spread': data.get('spread', 0).mean() if 'spread' in data else 0,
            'order_imbalance': data.get('order_imbalance', 0).mean() if 'order_imbalance' in data else 0,
            'trade_intensity': len(data) / (data.index[-1] - data.index[0]).total_seconds()
        }
        
    def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect market anomalies"""
        anomalies = []
        
        # Price spikes
        price_changes = data.groupby('symbol')['price'].pct_change()
        spike_threshold = price_changes.std() * 3
        spikes = price_changes[price_changes.abs() > spike_threshold]
        
        if not spikes.empty:
            anomalies.append({
                'type': 'price_spike',
                'symbols': spikes.index.get_level_values('symbol').unique().tolist(),
                'magnitude': spikes.abs().max()
            })
            
        # Volume anomalies
        volume_mean = data.groupby('symbol')['volume'].mean()
        volume_std = data.groupby('symbol')['volume'].std()
        volume_zscore = (data['volume'] - volume_mean) / volume_std
        volume_anomalies = data[volume_zscore.abs() > 3]
        
        if not volume_anomalies.empty:
            anomalies.append({
                'type': 'volume_anomaly',
                'symbols': volume_anomalies['symbol'].unique().tolist(),
                'magnitude': volume_zscore.abs().max()
            })
            
        return anomalies
        
    def _analyze_sector_rotation(self, data: pd.DataFrame) -> Dict:
        """Analyze sector rotation patterns"""
        # Simplified sector analysis
        # In production, would map symbols to sectors
        return {
            'leading_sectors': ['Technology', 'Healthcare'],
            'lagging_sectors': ['Energy', 'Utilities'],
            'rotation_strength': 0.7
        }
        
    def _analyze_factor_performance(self, 
                                  existing_factors: pd.DataFrame,
                                  performance_metrics: Dict) -> Dict:
        """Analyze existing factor performance"""
        return {
            'top_factors': self._get_top_factors(performance_metrics),
            'declining_factors': self._get_declining_factors(performance_metrics),
            'factor_correlations': self._calculate_factor_correlations(existing_factors),
            'factor_decay_rates': self._calculate_decay_rates(performance_metrics)
        }
        
    def _get_top_factors(self, metrics: Dict) -> List[Dict]:
        """Get top performing factors"""
        # Extract top factors by IC
        factor_ics = metrics.get('factor_ic', {})
        sorted_factors = sorted(factor_ics.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'name': name, 'ic': ic, 'category': self._get_factor_category(name)}
            for name, ic in sorted_factors[:5]
        ]
        
    def _get_declining_factors(self, metrics: Dict) -> List[Dict]:
        """Get factors with declining performance"""
        # Identify factors with decreasing IC
        factor_trends = metrics.get('factor_ic_trend', {})
        declining = [
            {'name': name, 'decline_rate': trend}
            for name, trend in factor_trends.items()
            if trend < -0.1  # 10% decline
        ]
        return declining
        
    def _calculate_factor_correlations(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate factor correlation matrix"""
        if factors.empty:
            return pd.DataFrame()
        return factors.corr()
        
    def _calculate_decay_rates(self, metrics: Dict) -> Dict:
        """Calculate factor decay rates"""
        return metrics.get('factor_decay_rates', {})
        
    def _get_factor_category(self, factor_name: str) -> str:
        """Categorize factor by name"""
        if 'momentum' in factor_name.lower():
            return 'momentum'
        elif 'volatility' in factor_name.lower():
            return 'volatility'
        elif 'liquidity' in factor_name.lower() or 'volume' in factor_name.lower():
            return 'liquidity'
        elif 'microstructure' in factor_name.lower() or 'spread' in factor_name.lower():
            return 'microstructure'
        else:
            return 'other'
            
    async def _call_llm_for_factors(self, 
                                  market_analysis: Dict,
                                  factor_analysis: Dict) -> List[Dict]:
        """Call LLM API to generate new factors"""
        
        # Prepare prompts
        prompts = self._prepare_llm_prompts(market_analysis, factor_analysis)
        
        # Call LLM (example with local model API)
        new_factors = []
        for prompt_type, prompt in prompts.items():
            try:
                response = await self._make_llm_request(prompt)
                factors = self._parse_llm_response(response, prompt_type)
                new_factors.extend(factors)
            except Exception as e:
                logger.error(f"LLM request failed for {prompt_type}: {e}")
                
        return new_factors
        
    def _prepare_llm_prompts(self, 
                           market_analysis: Dict,
                           factor_analysis: Dict) -> Dict[str, str]:
        """Prepare prompts for LLM"""
        prompts = {}
        
        # Factor discovery prompt
        prompts['discovery'] = f"""
        Based on the following market analysis:
        
        Market Conditions:
        - Volatility Regime: {market_analysis['volatility_regime']}
        - Trend Strength: {json.dumps(market_analysis['trend_strength'], indent=2)}
        - Microstructure: {json.dumps(market_analysis['market_microstructure'], indent=2)}
        - Anomalies: {json.dumps(market_analysis['anomalies'], indent=2)}
        
        Current Factor Performance:
        - Top Factors: {json.dumps(factor_analysis['top_factors'], indent=2)}
        - Declining Factors: {json.dumps(factor_analysis['declining_factors'], indent=2)}
        
        Task: Generate 3 new alpha factors that could capture profit opportunities in the next 1-3 days.
        
        For each factor provide:
        1. Factor name (descriptive, e.g., "adaptive_volatility_momentum")
        2. Mathematical formula (using available data fields)
        3. Economic intuition (why this factor should predict returns)
        4. Required data fields
        5. Suggested parameters (e.g., lookback windows)
        6. Expected Information Coefficient (IC)
        
        Focus on factors that:
        - Are uncorrelated with existing top factors
        - Exploit the current market regime
        - Have clear economic rationale
        - Can predict 10%+ moves in 1-3 days
        
        Output format: JSON array of factor objects
        """
        
        # Factor combination prompt
        prompts['combination'] = f"""
        Given these existing factors with their performance:
        {json.dumps(factor_analysis['top_factors'], indent=2)}
        
        Create 2 non-linear combinations that:
        1. Maximize predictive power for 3-day 10% returns
        2. Maintain interpretability
        3. Reduce overfitting risk
        4. Account for factor interactions
        
        For each combination provide:
        1. Combination name
        2. Mathematical formula combining factors
        3. Rationale for the combination
        4. Expected improvement over individual factors
        
        Output format: JSON array
        """
        
        # Adaptive factor prompt
        prompts['adaptive'] = f"""
        Current market shows these anomalies:
        {json.dumps(market_analysis['anomalies'], indent=2)}
        
        Design an adaptive factor that:
        1. Automatically adjusts to market regimes
        2. Captures the specific anomaly patterns
        3. Has built-in risk controls
        
        Provide:
        1. Factor name
        2. Adaptive mechanism (how it adjusts)
        3. Mathematical formula
        4. Risk controls
        5. Backtest considerations
        
        Output format: JSON
        """
        
        return prompts
        
    async def _make_llm_request(self, prompt: str) -> str:
        """Make async request to LLM API"""
        # Example implementation for local LLM server
        url = "http://localhost:8000/v1/completions"  # Local LLM endpoint
        
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.llm_config['model'],
            "prompt": prompt,
            "temperature": self.llm_config['temperature'],
            "max_tokens": self.llm_config['max_tokens'],
            "response_format": {"type": "json_object"}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['text']
                else:
                    raise Exception(f"LLM request failed: {response.status}")
                    
    def _parse_llm_response(self, response: str, prompt_type: str) -> List[Dict]:
        """Parse LLM response into factor definitions"""
        try:
            # Parse JSON response
            data = json.loads(response)
            
            # Ensure it's a list
            if isinstance(data, dict):
                data = [data]
                
            # Validate and clean each factor
            factors = []
            for item in data:
                if self._validate_factor_definition(item):
                    factors.append(item)
                    
            return factors
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []
            
    def _validate_factor_definition(self, factor_def: Dict) -> bool:
        """Validate factor definition from LLM"""
        required_fields = ['name', 'formula', 'description']
        
        # Check required fields
        for field in required_fields:
            if field not in factor_def:
                logger.warning(f"Missing required field: {field}")
                return False
                
        # Validate formula syntax (basic check)
        try:
            # This is a simplified validation
            # In production, would use AST parsing
            formula = factor_def['formula']
            if not formula or len(formula) < 5:
                return False
                
            # Check for dangerous operations
            dangerous_ops = ['exec', 'eval', '__import__', 'open', 'file']
            for op in dangerous_ops:
                if op in formula:
                    logger.warning(f"Dangerous operation in formula: {op}")
                    return False
                    
        except Exception as e:
            logger.error(f"Formula validation error: {e}")
            return False
            
        return True
        
    def _validate_factors(self, 
                         factors: List[Dict], 
                         market_data: pd.DataFrame) -> List[GeneratedFactor]:
        """Validate and convert factors to GeneratedFactor objects"""
        validated = []
        
        for factor_def in factors:
            try:
                # Create GeneratedFactor object
                factor = GeneratedFactor(
                    name=factor_def.get('name', 'unnamed_factor'),
                    formula=factor_def.get('formula', ''),
                    description=factor_def.get('description', ''),
                    economic_intuition=factor_def.get('economic_intuition', ''),
                    required_data=factor_def.get('required_data', []),
                    parameters=factor_def.get('parameters', {}),
                    expected_ic=factor_def.get('expected_ic', 0.0),
                    confidence_score=self._calculate_confidence_score(factor_def),
                    generation_timestamp=datetime.now()
                )
                
                # Test factor calculation
                if self._test_factor_calculation(factor, market_data):
                    validated.append(factor)
                    logger.info(f"Validated new factor: {factor.name}")
                else:
                    logger.warning(f"Factor calculation failed: {factor.name}")
                    
            except Exception as e:
                logger.error(f"Factor validation error: {e}")
                
        return validated
        
    def _calculate_confidence_score(self, factor_def: Dict) -> float:
        """Calculate confidence score for generated factor"""
        score = 0.5  # Base score
        
        # Adjust based on completeness
        if factor_def.get('economic_intuition'):
            score += 0.1
        if factor_def.get('expected_ic', 0) > 0:
            score += 0.1
        if len(factor_def.get('required_data', [])) > 0:
            score += 0.1
        if factor_def.get('parameters'):
            score += 0.1
            
        # Cap at 1.0
        return min(score, 1.0)
        
    def _test_factor_calculation(self, 
                                factor: GeneratedFactor,
                                market_data: pd.DataFrame) -> bool:
        """Test if factor can be calculated with available data"""
        try:
            # Check if required data fields exist
            for field in factor.required_data:
                if field not in market_data.columns:
                    logger.warning(f"Missing required field: {field}")
                    return False
                    
            # Try to evaluate formula (simplified)
            # In production, would use safe expression evaluation
            sample_data = market_data.head(100)
            
            # This is a placeholder - actual implementation would
            # safely evaluate the formula
            return True
            
        except Exception as e:
            logger.error(f"Factor calculation test failed: {e}")
            return False
            
    def save_generated_factors(self, filepath: str):
        """Save generated factors to file"""
        factors_data = []
        
        for factor in self.generated_factors:
            factors_data.append({
                'name': factor.name,
                'formula': factor.formula,
                'description': factor.description,
                'economic_intuition': factor.economic_intuition,
                'required_data': factor.required_data,
                'parameters': factor.parameters,
                'expected_ic': factor.expected_ic,
                'confidence_score': factor.confidence_score,
                'generation_timestamp': factor.generation_timestamp.isoformat()
            })
            
        with open(filepath, 'w') as f:
            json.dump(factors_data, f, indent=2)
            
        logger.info(f"Saved {len(factors_data)} generated factors to {filepath}")