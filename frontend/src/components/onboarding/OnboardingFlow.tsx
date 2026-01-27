import React, { useState } from 'react';
import { connectWallet } from '../../utils/wallet';

interface OnboardingFlowProps {
  onComplete: () => void;
}

interface UserData {
  frequency: string;
  activities: string[];
  walletConnected: boolean;
  alertThreshold: number;
}

const OnboardingFlow: React.FC<OnboardingFlowProps> = ({ onComplete }) => {
  const [step, setStep] = useState(1);
  const [userData, setUserData] = useState<UserData>({
    frequency: '',
    activities: [],
    walletConnected: false,
    alertThreshold: 0.0025
  });

  const steps = [
    {
      title: "Welcome to Base Gas Optimiser! 👋",
      component: <WelcomeStep />,
      description: "Let's get you set up to start saving money. This will take less than 2 minutes."
    },
    {
      title: "Tell us about your usage",
      component: <UsageStep userData={userData} setUserData={setUserData} />,
      description: "Help us personalize your experience"
    },
    {
      title: "Connect your wallet (Optional)",
      component: <WalletStep userData={userData} setUserData={setUserData} />,
      description: "Get personalized insights and transaction history"
    },
    {
      title: "Set your first alert",
      component: <AlertStep userData={userData} setUserData={setUserData} />,
      description: "Get notified when gas prices drop"
    },
    {
      title: "You're all set! 🎉",
      component: <SuccessStep />,
      description: "Start saving money on your Base transactions"
    }
  ];

  const handleComplete = () => {
    localStorage.setItem('onboarding_completed', 'true');
    localStorage.setItem('user_preferences', JSON.stringify(userData));
    onComplete();
  };

  return (
    <div className="fixed inset-0 bg-gray-900/95 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-800 border border-gray-700 rounded-xl max-w-2xl w-full p-8">
        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-400">Step {step} of {steps.length}</span>
            <span className="text-sm text-gray-400">{Math.round((step / steps.length) * 100)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-cyan-500 to-emerald-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(step / steps.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Content */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">{steps[step - 1].title}</h2>
          <p className="text-gray-400">{steps[step - 1].description}</p>
        </div>

        {/* Step component */}
        <div className="mb-8">
          {steps[step - 1].component}
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          {step > 1 && (
            <button
              onClick={() => setStep(step - 1)}
              className="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
            >
              ← Back
            </button>
          )}
          
          {step < steps.length ? (
            <button
              onClick={() => setStep(step + 1)}
              className="ml-auto px-8 py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
            >
              Continue →
            </button>
          ) : (
            <button
              onClick={handleComplete}
              className="ml-auto px-8 py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
            >
              Go to Dashboard →
            </button>
          )}
        </div>
        
        {/* Skip option */}
        <div className="text-center mt-6">
          <button
            onClick={onComplete}
            className="text-gray-400 hover:text-white text-sm transition"
          >
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
};

const WelcomeStep = () => (
  <div className="text-center py-8">
    <div className="text-6xl mb-6">⚡</div>
    <p className="text-lg text-gray-300">
      Base Gas Optimiser uses AI to predict optimal transaction times,
      helping you save up to 65% on gas fees.
    </p>
  </div>
);

interface UsageStepProps {
  userData: UserData;
  setUserData: (data: UserData) => void;
}

const UsageStep: React.FC<UsageStepProps> = ({ userData, setUserData }) => (
  <div className="space-y-6">
    <div>
      <label className="block text-white font-medium mb-3">
        How often do you transact on Base?
      </label>
      <div className="space-y-2">
        {[
          { value: 'daily', label: 'Daily (5+ transactions/week)' },
          { value: 'weekly', label: 'Weekly (2-5 transactions/week)' },
          { value: 'monthly', label: 'Monthly (< 2 transactions/week)' }
        ].map(option => (
          <label key={option.value} className="flex items-center p-4 bg-gray-700/50 rounded-lg cursor-pointer hover:bg-gray-700 transition">
            <input
              type="radio"
              name="frequency"
              value={option.value}
              checked={userData.frequency === option.value}
              onChange={(e) => setUserData({ ...userData, frequency: e.target.value })}
              className="mr-3"
            />
            <span className="text-white">{option.label}</span>
          </label>
        ))}
      </div>
    </div>

    <div>
      <label className="block text-white font-medium mb-3">
        What do you mainly do? (Select all that apply)
      </label>
      <div className="space-y-2">
        {['DeFi Trading', 'NFT Minting', 'Token Transfers', 'Other'].map(activity => (
          <label key={activity} className="flex items-center p-4 bg-gray-700/50 rounded-lg cursor-pointer hover:bg-gray-700 transition">
            <input
              type="checkbox"
              checked={userData.activities.includes(activity)}
              onChange={(e) => {
                if (e.target.checked) {
                  setUserData({ ...userData, activities: [...userData.activities, activity] });
                } else {
                  setUserData({ ...userData, activities: userData.activities.filter(a => a !== activity) });
                }
              }}
              className="mr-3"
            />
            <span className="text-white">{activity}</span>
          </label>
        ))}
      </div>
    </div>
  </div>
);

interface WalletStepProps {
  userData: UserData;
  setUserData: (data: UserData) => void;
}

const WalletStep: React.FC<WalletStepProps> = ({ userData, setUserData }) => {
  const [connecting, setConnecting] = useState(false);

  const handleConnect = async () => {
    try {
      setConnecting(true);
      await connectWallet();
      setUserData({ ...userData, walletConnected: true });
    } catch (error: any) {
      alert(error.message || 'Failed to connect wallet');
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className="text-center py-8 space-y-6">
      <p className="text-gray-300">
        Connect your wallet to unlock personalized insights and see your transaction history
      </p>
      
      <div className="space-y-3">
        <button 
          onClick={handleConnect}
          disabled={connecting || userData.walletConnected}
          className="w-full p-4 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg font-semibold hover:shadow-lg transition flex items-center justify-center disabled:opacity-50"
        >
          <span className="mr-2">🦊</span> 
          {connecting ? 'Connecting...' : userData.walletConnected ? '✓ Connected' : 'Connect MetaMask'}
        </button>
        <button 
          onClick={() => setUserData({ ...userData, walletConnected: false })}
          className="w-full p-4 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
        >
          Skip for now
        </button>
      </div>
    </div>
  );
};

interface AlertStepProps {
  userData: UserData;
  setUserData: (data: UserData) => void;
}

const AlertStep: React.FC<AlertStepProps> = ({ userData, setUserData }) => (
  <div className="space-y-6">
    <div>
      <label className="block text-white font-medium mb-3">
        Alert me when Base gas drops below:
      </label>
      <div className="flex items-center space-x-4">
        <input
          type="number"
          step="0.0001"
          value={userData.alertThreshold}
          onChange={(e) => setUserData({ ...userData, alertThreshold: parseFloat(e.target.value) || 0.0025 })}
          className="flex-1 px-4 py-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-cyan-500 focus:outline-none"
        />
        <span className="text-gray-400">gwei</span>
      </div>
      <p className="text-sm text-gray-400 mt-2">
        Current gas: 0.0033 gwei (you'll be alerted when it drops 24%)
      </p>
    </div>

    <div className="bg-gray-700/50 border border-cyan-500/30 rounded-lg p-4">
      <div className="flex items-start">
        <span className="text-2xl mr-3">💡</span>
        <div>
          <p className="text-white font-medium mb-1">Tip: Enable browser notifications</p>
          <p className="text-sm text-gray-400">
            We'll notify you instantly when gas prices drop so you never miss optimal transaction times
          </p>
        </div>
      </div>
    </div>

    <button 
      onClick={async () => {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          alert('Notifications enabled!');
        }
      }}
      className="w-full p-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
    >
      🔔 Enable Notifications
    </button>
  </div>
);

const SuccessStep = () => (
  <div className="text-center py-8 space-y-6">
    <div className="text-6xl">🎉</div>
    <h3 className="text-2xl font-bold text-white">You're all set!</h3>
    
    <div className="bg-gray-700/50 rounded-lg p-6 text-left">
      <p className="text-white font-medium mb-4">Here's what happens next:</p>
      <ul className="space-y-3">
        <li className="flex items-start">
          <span className="text-green-400 mr-3">✅</span>
          <span className="text-gray-300">We're monitoring Base gas prices 24/7</span>
        </li>
        <li className="flex items-start">
          <span className="text-green-400 mr-3">✅</span>
          <span className="text-gray-300">You'll get alerts when gas drops below your threshold</span>
        </li>
        <li className="flex items-start">
          <span className="text-green-400 mr-3">✅</span>
          <span className="text-gray-300">Check predictions anytime in your dashboard</span>
        </li>
        <li className="flex items-start">
          <span className="text-green-400 mr-3">✅</span>
          <span className="text-gray-300">Track your savings over time</span>
        </li>
      </ul>
    </div>

    <p className="text-gray-400">
      Pro tip: Check the dashboard daily to maximize your savings!
    </p>
  </div>
);

export default OnboardingFlow;
