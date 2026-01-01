import numpy as np
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

# Try to import anthropic, but make it optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not installed. LLM explanations will use fallback.")


class PredictionExplainer:
    """
    Explains ML predictions by showing feature importance and similar historical cases
    """
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.anthropic_client = None
        
        # Initialize Anthropic client if available and API key is set
        if ANTHROPIC_AVAILABLE:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                try:
                    self.anthropic_client = anthropic.Anthropic(api_key=api_key)
                    logger.info("Anthropic client initialized for LLM explanations")
                except Exception as e:
                    logger.warning(f"Failed to initialize Anthropic client: {e}")
            else:
                logger.info("ANTHROPIC_API_KEY not set, using fallback explanations")
    
    def explain_prediction(self, X, prediction, historical_data, current_gas=None):
        """
        Generate explanation for a prediction with LLM-powered natural language
        
        Args:
            X: Feature vector for prediction
            prediction: The predicted gas price
            historical_data: Historical gas price data
            current_gas: Current gas price (for LLM context)
        
        Returns:
            dict with LLM explanation and technical details
        """
        
        # 1. Feature importance
        feature_importance = self._calculate_feature_importance(X)
        
        # 2. Find similar historical situations
        similar_cases = self._find_similar_cases(X, historical_data)
        
        # 3. Determine factors increasing/decreasing gas
        increasing_factors, decreasing_factors = self._categorize_factors(
            X, feature_importance
        )
        
        # 4. Generate LLM explanation (NEW)
        if current_gas is None:
            # Try to extract from X or use a default
            current_gas = float(prediction) * 0.95  # Estimate
        
        llm_explanation = self._generate_llm_explanation(
            current_gas=current_gas,
            prediction=prediction,
            increasing_factors=increasing_factors,
            decreasing_factors=decreasing_factors,
            similar_cases=similar_cases,
            feature_data=X
        )
        
        # 5. Generate technical explanation (fallback/legacy)
        technical_explanation = self._generate_explanation_text(
            increasing_factors,
            decreasing_factors,
            similar_cases
        )
        
        return {
            'llm_explanation': llm_explanation,  # Simple, conversational explanation
            'technical_explanation': technical_explanation,  # Legacy format
            'technical_details': {  # Technical data (shown on "Show Details")
                'feature_importance': feature_importance,
                'increasing_factors': increasing_factors,
                'decreasing_factors': decreasing_factors,
                'similar_cases': similar_cases
            },
            'prediction': float(prediction),
            'current_gas': float(current_gas)
        }
    
    def _calculate_feature_importance(self, X):
        """Calculate importance of each feature for this prediction"""
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X_array = X.values
        elif hasattr(X, 'iloc'):
            X_array = X.iloc[0:1].values
        else:
            X_array = np.array(X)
        
        # Ensure 2D array
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)
        
        # Simple method: compare prediction when each feature is set to mean
        try:
            base_pred = self.model.predict(X_array)[0]
        except:
            base_pred = self.model.predict(X_array.reshape(1, -1))[0]
        
        importances = {}
        
        for i, feature_name in enumerate(self.feature_names):
            try:
                if i >= X_array.shape[1]:
                    continue
                    
                # Create modified feature vector
                X_modified = X_array.copy()
                X_modified[0, i] = 0  # Set feature to baseline
                
                # Predict with modified features
                try:
                    modified_pred = self.model.predict(X_modified)[0]
                except:
                    modified_pred = self.model.predict(X_modified.reshape(1, -1))[0]
                
                # Importance = change in prediction
                importance = abs(base_pred - modified_pred)
                
                importances[feature_name] = {
                    'value': float(X_array[0, i]),
                    'importance': float(importance),
                    'impact': 'increase' if (base_pred - modified_pred) > 0 else 'decrease'
                }
            except Exception as e:
                logger.warning(f"Error calculating importance for {feature_name}: {e}")
                try:
                    importances[feature_name] = {
                        'value': float(X_array[0, i]) if i < X_array.shape[1] else 0,
                        'importance': 0.0,
                        'impact': 'neutral'
                    }
                except:
                    importances[feature_name] = {
                        'value': 0.0,
                        'importance': 0.0,
                        'impact': 'neutral'
                    }
        
        # Sort by importance
        sorted_features = sorted(
            importances.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )
        
        return dict(sorted_features)
    
    def _find_similar_cases(self, X, historical_data, n=3):
        """Find n most similar historical situations"""
        
        if not historical_data or len(historical_data) < 5:
            return []
        
        # Extract features from current prediction
        try:
            # Convert X to array if needed
            if hasattr(X, 'values'):
                X_array = X.values
            elif hasattr(X, 'iloc'):
                X_array = X.iloc[0:1].values
            else:
                X_array = np.array(X)
            
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)
            
            hour_idx = self.feature_names.index('hour') if 'hour' in self.feature_names else None
            day_idx = self.feature_names.index('day_of_week') if 'day_of_week' in self.feature_names else None
            
            current_hour = int(X_array[0, hour_idx]) if hour_idx is not None and hour_idx < X_array.shape[1] else datetime.now().hour
            current_day = int(X_array[0, day_idx]) if day_idx is not None and day_idx < X_array.shape[1] else datetime.now().weekday()
        except:
            current_hour = datetime.now().hour
            current_day = datetime.now().weekday()
        
        similar = []
        
        for data in historical_data:
            try:
                timestamp_str = data.get('timestamp', '')
                if isinstance(timestamp_str, str):
                    from dateutil import parser
                    try:
                        timestamp = parser.parse(timestamp_str)
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = timestamp_str if hasattr(timestamp_str, 'hour') else datetime.now()
                
                gas_price = data.get('gas_price', 0) or data.get('current_gas', 0) or data.get('gwei', 0)
                
                if gas_price == 0:
                    continue
                
                # Check if hour and day match
                hour_match = timestamp.hour == current_hour if hasattr(timestamp, 'hour') else False
                day_match = timestamp.weekday() == current_day if hasattr(timestamp, 'weekday') else False
                
                if hour_match and day_match:
                    similar.append({
                        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(timestamp, 'strftime') else str(timestamp),
                        'gas_price': float(gas_price),
                        'similarity': 1.0 if (hour_match and day_match) else 0.5
                    })
            except Exception as e:
                logger.warning(f"Error processing historical case: {e}")
                continue
        
        # Sort by similarity and recency
        similar.sort(key=lambda x: (x['similarity'], x['timestamp']), reverse=True)
        
        return similar[:n]
    
    def _categorize_factors(self, X, feature_importance):
        """Categorize features into increasing vs decreasing gas"""
        
        increasing = []
        decreasing = []
        
        for feature_name, data in feature_importance.items():
            if data['importance'] < 0.0001:  # Skip insignificant features
                continue
            
            # Create human-readable description
            description = self._describe_feature(feature_name, data['value'])
            
            weight_percent = min(100, int(data['importance'] * 1000))  # Scale for display
            
            factor = {
                'name': feature_name,
                'description': description,
                'weight': weight_percent,
                'value': data['value']
            }
            
            if data['impact'] == 'increase':
                increasing.append(factor)
            else:
                decreasing.append(factor)
        
        return increasing, decreasing
    
    def _describe_feature(self, feature_name, value):
        """Convert feature to human-readable description"""
        
        descriptions = {
            'hour': {
                'name': 'Time of day',
                'values': {
                    tuple(range(0, 6)): 'Late night (low activity)',
                    tuple(range(6, 12)): 'Morning (increasing activity)',
                    tuple(range(12, 18)): 'Afternoon (peak hours)',
                    tuple(range(18, 24)): 'Evening (declining activity)'
                }
            },
            'day_of_week': {
                'name': 'Day of week',
                'values': {
                    (5, 6): 'Weekend (low activity)',
                    (0, 1, 2, 3, 4): 'Weekday (higher activity)'
                }
            },
            'trend_1h': {
                'name': 'Recent 1h trend',
                'condition': lambda v: 'Upward trend' if v > 0 else 'Downward trend'
            },
            'trend_3h': {
                'name': 'Recent 3h trend',
                'condition': lambda v: 'Upward trend' if v > 0 else 'Downward trend'
            }
        }
        
        if feature_name in descriptions:
            desc = descriptions[feature_name]
            
            if 'values' in desc:
                for value_range, text in desc['values'].items():
                    if isinstance(value_range, tuple):
                        if int(value) in value_range:
                            return f"{desc['name']}: {text}"
            
            if 'condition' in desc:
                return f"{desc['name']}: {desc['condition'](value)}"
        
        return f"{feature_name}: {value:.4f}"
    
    def _generate_llm_explanation(self, current_gas, prediction, increasing_factors, 
                                   decreasing_factors, similar_cases, feature_data):
        """
        Use Claude API to generate natural language explanation
        Falls back to template-based explanation if LLM unavailable
        """
        
        if self.anthropic_client is None:
            # Use fallback explanation
            return self._generate_fallback_explanation(
                current_gas, prediction, increasing_factors, decreasing_factors, similar_cases
            )
        
        try:
            # Extract key information for LLM
            change_percent = ((prediction - current_gas) / current_gas) * 100 if current_gas > 0 else 0
            direction = "DROP" if prediction < current_gas else "RISE" if prediction > current_gas else "STAY STABLE"
            
            # Get most important factors
            top_increasing = increasing_factors[:2] if increasing_factors else []
            top_decreasing = decreasing_factors[:2] if decreasing_factors else []
            
            # Get feature values for context
            feature_context = self._extract_feature_context(feature_data)
            
            # Build prompt for Claude
            prompt = f"""You are explaining a gas price prediction to a blockchain user in simple, conversational language.

Current Base network gas price: {current_gas:.4f} gwei
Predicted gas price: {prediction:.4f} gwei
Expected change: {abs(change_percent):.1f}% {direction}

Context:
- Time: {feature_context.get('time_description', 'Unknown')}
- Day: {feature_context.get('day_description', 'Unknown')}
- Recent trend: {feature_context.get('trend_description', 'Unknown')}

Factors DECREASING gas (pushing price down):
{self._format_factors_for_llm(top_decreasing)}

Factors INCREASING gas (pushing price up):
{self._format_factors_for_llm(top_increasing)}

Similar historical situations averaged: {similar_cases[0]['gas_price']:.4f} gwei (from {len(similar_cases)} past occurrences) if similar_cases else 'No similar cases found'

Write a clear, friendly 2-3 sentence explanation of WHY gas is expected to {direction.lower()}. 

Requirements:
- Use simple language (avoid technical jargon)
- Be conversational and friendly
- Focus on the MAIN reason (the strongest factor)
- Include the percentage change
- Don't use words like "model", "algorithm", "ML", or "prediction"
- Just explain what's happening and why

Example good response:
"Gas is expected to drop about 35% because it's Saturday night when Base network activity is typically very low. Historically, we see similar low prices at this time, averaging around 0.0019 gwei."

Example bad response:
"Our machine learning model predicts a decrease based on multiple algorithmic factors including temporal features and network congestion metrics."

Your explanation:"""

            # Call Claude API
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract text from response
            explanation_text = message.content[0].text.strip()
            
            logger.info(f"Generated LLM explanation: {explanation_text}")
            
            return explanation_text
            
        except Exception as e:
            logger.error(f"Error generating LLM explanation: {e}")
            
            # Fallback to simple template-based explanation
            return self._generate_fallback_explanation(
                current_gas, prediction, increasing_factors, decreasing_factors, similar_cases
            )
    
    def _extract_feature_context(self, X):
        """Extract human-readable context from features"""
        context = {}
        
        # Convert X to array if needed
        try:
            if hasattr(X, 'values'):
                X_array = X.values
            elif hasattr(X, 'iloc'):
                X_array = X.iloc[0:1].values
            else:
                X_array = np.array(X)
            
            if len(X_array.shape) == 1:
                X_array = X_array.reshape(1, -1)
        except:
            return context
        
        # Get feature indices
        try:
            hour_idx = self.feature_names.index('hour') if 'hour' in self.feature_names else None
            day_idx = self.feature_names.index('day_of_week') if 'day_of_week' in self.feature_names else None
            is_weekend_idx = self.feature_names.index('is_weekend') if 'is_weekend' in self.feature_names else None
            
            # Time of day
            if hour_idx is not None and hour_idx < X_array.shape[1]:
                hour = int(X_array[0, hour_idx])
                if 0 <= hour < 6:
                    context['time_description'] = f"Late night ({hour}:00 UTC)"
                elif 6 <= hour < 12:
                    context['time_description'] = f"Morning ({hour}:00 UTC)"
                elif 12 <= hour < 18:
                    context['time_description'] = f"Afternoon ({hour}:00 UTC)"
                else:
                    context['time_description'] = f"Evening ({hour}:00 UTC)"
            
            # Day of week
            if is_weekend_idx is not None and is_weekend_idx < X_array.shape[1] and X_array[0, is_weekend_idx] == 1:
                context['day_description'] = "Weekend"
            elif day_idx is not None and day_idx < X_array.shape[1]:
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day = int(X_array[0, day_idx])
                context['day_description'] = days[day] if 0 <= day < 7 else "Unknown"
            
            # Trend
            trend_idx = self.feature_names.index('trend_1h') if 'trend_1h' in self.feature_names else None
            if trend_idx is not None and trend_idx < X_array.shape[1]:
                trend = X_array[0, trend_idx]
                if trend > 0.0001:
                    context['trend_description'] = "Gas has been rising recently"
                elif trend < -0.0001:
                    context['trend_description'] = "Gas has been falling recently"
                else:
                    context['trend_description'] = "Gas has been stable"
            
        except Exception as e:
            logger.error(f"Error extracting feature context: {e}")
        
        return context
    
    def _format_factors_for_llm(self, factors):
        """Format factors for LLM prompt"""
        if not factors:
            return "None"
        
        formatted = []
        for factor in factors:
            # Clean up description for LLM
            desc = factor['description']
            # Remove technical jargon
            desc = desc.replace('is_weekend:', 'Weekend:')
            desc = desc.replace('hour_cos:', 'Time of day:')
            desc = desc.replace('gas_change_1h:', 'Recent change:')
            formatted.append(f"- {desc} (weight: {factor['weight']}%)")
        
        return "\n".join(formatted)
    
    def _generate_fallback_explanation(self, current_gas, prediction, increasing_factors, 
                                       decreasing_factors, similar_cases):
        """
        Fallback explanation if LLM call fails or unavailable
        """
        
        change_percent = ((prediction - current_gas) / current_gas) * 100 if current_gas > 0 else 0
        direction = "DROP" if prediction < current_gas else "RISE" if prediction > current_gas else "STAY STABLE"
        
        if direction == "DROP" and decreasing_factors:
            main_reason = decreasing_factors[0]['description']
            # Clean up the reason
            main_reason = main_reason.replace('is_weekend: 1.0000', 'it\'s the weekend')
            main_reason = main_reason.replace('hour_cos:', 'time of day')
            main_reason = main_reason.replace('gas_change_1h:', 'recent trend')
            
            avg_gas = similar_cases[0]['gas_price'] if similar_cases else prediction
            return f"Gas is expected to drop about {abs(change_percent):.0f}% mainly because {main_reason.lower()}. Historically, similar situations show gas around {avg_gas:.4f} gwei."
        
        elif direction == "RISE" and increasing_factors:
            main_reason = increasing_factors[0]['description']
            main_reason = main_reason.replace('is_weekend: 0.0000', 'it\'s a weekday')
            main_reason = main_reason.replace('hour_cos:', 'time of day')
            
            return f"Gas is expected to rise about {abs(change_percent):.0f}% mainly because {main_reason.lower()}. You might want to transact soon to avoid higher costs."
        
        else:
            return f"Gas is expected to stay relatively stable around {prediction:.4f} gwei, changing only about {abs(change_percent):.0f}%."
    
    def _generate_explanation_text(self, increasing, decreasing, similar):
        """Generate human-readable explanation (legacy method)"""
        
        text = ""
        
        if decreasing and len(decreasing) > len(increasing):
            text += "Gas is expected to DROP because:\n"
            if decreasing:
                text += f"• {decreasing[0]['description']}\n"
            if len(decreasing) > 1:
                text += f"• {decreasing[1]['description']}\n"
        elif increasing:
            text += "Gas is expected to RISE because:\n"
            if increasing:
                text += f"• {increasing[0]['description']}\n"
            if len(increasing) > 1:
                text += f"• {increasing[1]['description']}\n"
        
        if similar:
            avg_gas = np.mean([s['gas_price'] for s in similar])
            text += f"\nSimilar situations historically: {avg_gas:.4f} gwei (average)"
        
        return text


# Global instance
explainer = None


def initialize_explainer(model, feature_names):
    global explainer
    explainer = PredictionExplainer(model, feature_names)
    return explainer

