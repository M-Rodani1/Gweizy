import React, { useState, useEffect, useCallback } from 'react';
import { X, ChevronRight, ChevronLeft, Sparkles, Bot, BarChart3, Settings2, Zap } from 'lucide-react';

interface TourStep {
  id: string;
  title: string;
  description: string;
  target: string; // CSS selector or element ID
  position: 'top' | 'bottom' | 'left' | 'right';
  icon: React.ReactNode;
}

const TOUR_STEPS: TourStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to Gweizy!',
    description: 'Your AI-powered gas optimisation assistant. Let me show you around.',
    target: 'body',
    position: 'bottom',
    icon: <Sparkles className="w-5 h-5 text-cyan-400" />
  },
  {
    id: 'pilot',
    title: 'AI Transaction Pilot',
    description: 'This is your command center. Get real-time recommendations on when to execute transactions based on gas prices.',
    target: '[data-tour="pilot"]',
    position: 'bottom',
    icon: <Bot className="w-5 h-5 text-cyan-400" />
  },
  {
    id: 'tabs',
    title: 'Dashboard Tabs',
    description: 'Switch between Overview, Analytics, and System tabs to access different features.',
    target: '[data-tour="tabs"]',
    position: 'bottom',
    icon: <BarChart3 className="w-5 h-5 text-purple-400" />
  },
  {
    id: 'forecast',
    title: 'Price Forecast',
    description: 'See predicted gas prices for the next 1, 4, and 24 hours to plan your transactions.',
    target: '[data-tour="forecast"]',
    position: 'right',
    icon: <Zap className="w-5 h-5 text-amber-400" />
  },
  {
    id: 'profile',
    title: 'Your Profile',
    description: 'Customise your strategy, default transaction type, and urgency preferences here.',
    target: '[data-tour="profile"]',
    position: 'top',
    icon: <Settings2 className="w-5 h-5 text-emerald-400" />
  }
];

const STORAGE_KEY = 'gweizy_tour_completed';

interface OnboardingTourProps {
  onComplete?: () => void;
  forceShow?: boolean;
}

const OnboardingTour: React.FC<OnboardingTourProps> = ({ onComplete, forceShow = false }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);

  // Check if tour should be shown
  useEffect(() => {
    if (forceShow) {
      setIsVisible(true);
      return;
    }

    const hasCompleted = localStorage.getItem(STORAGE_KEY);
    if (!hasCompleted) {
      // Small delay to let the page render
      const timer = setTimeout(() => setIsVisible(true), 1000);
      return () => clearTimeout(timer);
    }
  }, [forceShow]);

  // Update target element position
  const updateTargetPosition = useCallback(() => {
    const step = TOUR_STEPS[currentStep];
    if (step.target === 'body') {
      setTargetRect(null);
      return;
    }

    const element = document.querySelector(step.target);
    if (element) {
      const rect = element.getBoundingClientRect();
      setTargetRect(rect);

      // Scroll element into view if needed
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [currentStep]);

  useEffect(() => {
    if (isVisible) {
      updateTargetPosition();
      window.addEventListener('resize', updateTargetPosition);
      window.addEventListener('scroll', updateTargetPosition);
      return () => {
        window.removeEventListener('resize', updateTargetPosition);
        window.removeEventListener('scroll', updateTargetPosition);
      };
    }
  }, [isVisible, currentStep, updateTargetPosition]);

  const handleNext = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = () => {
    localStorage.setItem(STORAGE_KEY, 'true');
    setIsVisible(false);
    onComplete?.();
  };

  const handleSkip = () => {
    localStorage.setItem(STORAGE_KEY, 'true');
    setIsVisible(false);
    onComplete?.();
  };

  if (!isVisible) return null;

  const step = TOUR_STEPS[currentStep];
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === TOUR_STEPS.length - 1;

  // Calculate tooltip position
  const getTooltipStyle = (): React.CSSProperties => {
    if (!targetRect || step.target === 'body') {
      // Center on screen for welcome step
      return {
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)'
      };
    }

    const padding = 16;
    const tooltipWidth = 320;
    const tooltipHeight = 200;

    switch (step.position) {
      case 'bottom':
        return {
          position: 'fixed',
          top: targetRect.bottom + padding,
          left: Math.max(padding, Math.min(targetRect.left + targetRect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - padding))
        };
      case 'top':
        return {
          position: 'fixed',
          top: targetRect.top - tooltipHeight - padding,
          left: Math.max(padding, Math.min(targetRect.left + targetRect.width / 2 - tooltipWidth / 2, window.innerWidth - tooltipWidth - padding))
        };
      case 'left':
        return {
          position: 'fixed',
          top: targetRect.top + targetRect.height / 2 - tooltipHeight / 2,
          left: targetRect.left - tooltipWidth - padding
        };
      case 'right':
        return {
          position: 'fixed',
          top: targetRect.top + targetRect.height / 2 - tooltipHeight / 2,
          left: targetRect.right + padding
        };
      default:
        return {
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)'
        };
    }
  };

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/70 z-[9998] transition-opacity duration-300"
        onClick={handleSkip}
      />

      {/* Highlight box around target */}
      {targetRect && step.target !== 'body' && (
        <div
          className="fixed z-[9999] pointer-events-none"
          style={{
            top: targetRect.top - 8,
            left: targetRect.left - 8,
            width: targetRect.width + 16,
            height: targetRect.height + 16,
            boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.7), 0 0 20px rgba(6, 182, 212, 0.5)',
            borderRadius: '12px',
            border: '2px solid rgba(6, 182, 212, 0.5)'
          }}
        />
      )}

      {/* Tooltip */}
      <div
        className="z-[10000] w-80 bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl animate-fadeIn"
        style={getTooltipStyle()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gray-800 rounded-lg">
              {step.icon}
            </div>
            <div>
              <h3 className="font-semibold text-white">{step.title}</h3>
              <p className="text-xs text-gray-500">Step {currentStep + 1} of {TOUR_STEPS.length}</p>
            </div>
          </div>
          <button
            onClick={handleSkip}
            className="p-1 text-gray-500 hover:text-white transition-colors"
            aria-label="Close tour"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          <p className="text-gray-300 text-sm leading-relaxed">{step.description}</p>
        </div>

        {/* Progress bar */}
        <div className="px-4 pb-2">
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
              style={{ width: `${((currentStep + 1) / TOUR_STEPS.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-800">
          <button
            onClick={handleSkip}
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors"
          >
            Skip tour
          </button>
          <div className="flex items-center gap-2">
            {!isFirstStep && (
              <button
                onClick={handlePrev}
                className="flex items-center gap-1 px-3 py-2 text-sm text-gray-300 hover:text-white transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>
            )}
            <button
              onClick={handleNext}
              className="flex items-center gap-1 px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {isLastStep ? 'Get Started' : 'Next'}
              {!isLastStep && <ChevronRight className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default OnboardingTour;

// Hook to manually trigger tour
export function useTour() {
  const resetTour = () => {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  };

  const hasSeen = () => {
    return localStorage.getItem(STORAGE_KEY) === 'true';
  };

  return { resetTour, hasSeen };
}
