"""
ToneClassification.py
Classifies skin undertone based on LAB color space analysis
"""

import numpy as np


class ToneClassification:
    def __init__(self):
        """Initialize tone classification module"""
        pass
    
    def classify_undertone(self, tone_vector_lab):
        """
        Classify skin undertone based on LAB values
        
        Args:
            tone_vector_lab: dict with L, a, b values
        
        Returns:
            dict with undertone classification and confidence
        """
        L = tone_vector_lab["L"]
        a = tone_vector_lab["a"]
        b = tone_vector_lab["b"]
        
        print(f"\nAnalyzing undertone from LAB: L={L:.1f}, a={a:.1f}, b={b:.1f}")
        
        # Undertone classification based on a and b channels
        # a: green (-) to red (+)
        # b: blue (-) to yellow (+)
        
        undertone = None
        confidence = 0.0
        explanation = ""
        
        # Calculate chroma angle in a-b plane
        chroma_angle = np.degrees(np.arctan2(b, a))
        
        # Normalize to 0-360
        if chroma_angle < 0:
            chroma_angle += 360
        
        # Classification thresholds
        if b > 5 and a < 10:
            # High b (yellow), low a (less red) = warm/yellow undertone
            undertone = "warm"
            confidence = min(100, (b / 20) * 100)
            explanation = "High yellow (b) component indicates warm undertone"
        elif b < -2:
            # Negative b (blue) = cool/pink undertone
            undertone = "cool"
            confidence = min(100, abs(b / 10) * 100)
            explanation = "Blue (negative b) component indicates cool undertone"
        elif abs(b) <= 5 and abs(a) <= 10:
            # Balanced a and b = neutral undertone
            undertone = "neutral"
            confidence = 70.0
            explanation = "Balanced a and b values indicate neutral undertone"
        else:
            # Default to warm if b is positive
            if b > 0:
                undertone = "warm"
                confidence = 60.0
                explanation = "Slight yellow bias suggests warm undertone"
            else:
                undertone = "neutral"
                confidence = 50.0
                explanation = "Ambiguous values, classified as neutral"
        
        print(f"Undertone: {undertone.upper()} (confidence: {confidence:.1f}%)")
        print(f"Explanation: {explanation}")
        print(f"Chroma angle: {chroma_angle:.1f}°")
        
        return {
            "undertone": undertone,
            "confidence": confidence,
            "explanation": explanation,
            "chroma_angle": chroma_angle,
            "lab_values": {
                "L": L,
                "a": a,
                "b": b
            }
        }


if __name__ == "__main__":
    print("ToneClassification module - use via Orchestrator")
